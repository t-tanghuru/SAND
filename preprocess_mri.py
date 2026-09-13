"""Preprocess raw ADNI DICOM T1 volumes into 2D axial NIfTI slices ready for
`src/data/get_train_and_val_dataloader.py`.

Pipeline per subject/image:
  1. DICOM series -> 3D NIfTI volume (SimpleITK).
  2. N4 bias field correction.
  3. Skull-stripping with SynthStrip (Hoopes et al. 2022, NeuroImage), run via
     FreeSurfer's official `mri_synthstrip` script vendored verbatim under
     `third_party/` with the official `synthstrip.1.pt` weights (SHA256
     verified). `--skull_strip_method otsu` is a low-quality smoke-test
     fallback only -- never use it to produce training data.
  4. Rigid+affine registration to the MNI152 1mm brain template (ANTsPy).
  5. Resample to isotropic 1mm spacing.
  6. Robust intensity clipping (0.5-99.5 percentile inside the brain mask).
  7. Extract axial slices whose brain-mask coverage exceeds --min_brain_frac,
     save each as its own 2D NIfTI file (what the MONAI LoadImaged-based
     loader expects: one image file per dataset row).

Output layout:
  <output_dir>/nifti/<subject_id>_<image_id>.nii.gz       (intermediate 3D volume)
  <output_dir>/slices/<subject_id>_<image_id>_z<idx>.nii.gz (final 2D training data)
  <output_dir>/train_normal.csv, val_normal.csv, test_normal.csv
    (subject-level split; single row of comma-separated slice paths, matching
    the format `ddpm-ood`'s own get_*_dataset.py scripts write)

This only prepares the *normal* (CN) splits -- test_anomalous.csv needs a
separate lesion dataset (e.g. BRATS/WMH) and is out of scope here.
"""
import argparse
import csv
import hashlib
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

try:
    import ants

    HAS_ANTS = True
except ImportError:
    HAS_ANTS = False

SYNTHSTRIP_SCRIPT = Path(__file__).resolve().parent / "third_party" / "mri_synthstrip.py"
SYNTHSTRIP_MODEL_URL = "https://ftp.nmr.mgh.harvard.edu/pub/dist/freesurfer/synthstrip/models/synthstrip.1.pt"
# SHA256 recorded by the official freesurfer/freesurfer repo's git-annex key for synthstrip.1.pt.
SYNTHSTRIP_MODEL_SHA256 = "37417f802196186441aae3e7f385d94f8a98c64a88acaeaa2723af995c653e33"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dicom_root", required=True, help="Path to CN_MRI_raw_486subjects/")
    parser.add_argument("--output_dir", required=True, help="Where to write nifti/, slices/, and the split CSVs.")
    parser.add_argument("--mni_template", default=None, help="Path to an MNI152 T1 1mm brain-extracted template .nii.gz. If omitted, registration is skipped.")
    parser.add_argument("--skull_strip_method", choices=["synthstrip", "otsu"], default="synthstrip")
    parser.add_argument("--synthstrip_model", default=str(Path.home() / ".cache" / "synthstrip" / "synthstrip.1.pt"), help="Official SynthStrip weights; downloaded from MGH and SHA256-verified if missing.")
    parser.add_argument("--isotropic_spacing", type=float, default=1.0)
    parser.add_argument("--clip_low_pct", type=float, default=0.5)
    parser.add_argument("--clip_high_pct", type=float, default=99.5)
    parser.add_argument("--min_brain_frac", type=float, default=0.02, help="Skip axial slices where the brain mask covers less than this fraction of the slice.")
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--test_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--first_n", type=int, default=None, help="Only process the first N subjects (for a smoke test).")
    parser.add_argument("--gpu", action="store_true", help="Run SynthStrip on GPU.")
    return parser.parse_args()


def dicom_series_to_nifti(dicom_dir: Path) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(dicom_dir))
    if not series_ids:
        raise RuntimeError(f"No DICOM series found in {dicom_dir}")
    files = reader.GetGDCMSeriesFileNames(str(dicom_dir), series_ids[0])
    reader.SetFileNames(files)
    return reader.Execute()


def n4_bias_correct(image: sitk.Image) -> sitk.Image:
    image_f = sitk.Cast(image, sitk.sitkFloat32)
    mask = sitk.OtsuThreshold(image_f, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    return corrector.Execute(image_f, mask)


def skull_strip_otsu(image: sitk.Image) -> sitk.Image:
    """Fallback brain mask: Otsu threshold + largest connected component +
    morphological closing. Much less accurate than SynthStrip -- smoke-test only."""
    mask = sitk.OtsuThreshold(image, 0, 1, 200)
    mask = sitk.BinaryMorphologicalClosing(mask, [3, 3, 3])
    cc = sitk.ConnectedComponent(mask)
    cc = sitk.RelabelComponent(cc, sortByObjectSize=True)
    mask = sitk.Equal(cc, 1)
    mask = sitk.BinaryFillhole(mask)
    return sitk.Cast(mask, sitk.sitkUInt8)


def ensure_synthstrip_model(model_path: Path) -> Path:
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = model_path.with_suffix(".part")
        urllib.request.urlretrieve(SYNTHSTRIP_MODEL_URL, tmp)
        tmp.replace(model_path)
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if digest != SYNTHSTRIP_MODEL_SHA256:
        raise RuntimeError(f"{model_path} SHA256 {digest} does not match the official SynthStrip weights.")
    return model_path


def skull_strip_synthstrip(nifti_path: Path, out_dir: Path, model_path: Path, use_gpu: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = nifti_path.name.replace(".nii.gz", "")
    brain_path = out_dir / f"{stem}_brain.nii.gz"
    mask_path = out_dir / f"{stem}_mask.nii.gz"
    cmd = [sys.executable, str(SYNTHSTRIP_SCRIPT), "-i", str(nifti_path), "-o", str(brain_path), "-m", str(mask_path), "--model", str(model_path)]
    if use_gpu:
        cmd.append("-g")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"mri_synthstrip failed: {(result.stderr or result.stdout)[-2000:]}")
    return brain_path, mask_path


def register_to_template(brain_image: sitk.Image, template_path: str) -> sitk.Image:
    if not HAS_ANTS:
        raise RuntimeError("antspyx is not installed (`pip install antspyx`) -- required for --mni_template registration.")
    with tempfile.TemporaryDirectory() as tmp_dir:
        moving_path = Path(tmp_dir) / "moving.nii.gz"
        result_path = Path(tmp_dir) / "result.nii.gz"
        sitk.WriteImage(brain_image, str(moving_path))
        moving = ants.image_read(str(moving_path))
        fixed = ants.image_read(template_path)
        result = ants.registration(fixed=fixed, moving=moving, type_of_transform="Affine")
        ants.image_write(result["warpedmovout"], str(result_path))
        return sitk.ReadImage(str(result_path))


def resample_isotropic(image: sitk.Image, spacing: float, is_mask: bool = False) -> sitk.Image:
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_spacing = [spacing] * 3
    new_size = [
        int(round(osz * ospc / spacing)) for osz, ospc in zip(original_size, original_spacing)
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_mask else sitk.sitkBSpline)
    return resampler.Execute(image)


def clip_intensity(image: sitk.Image, mask: sitk.Image, low_pct: float, high_pct: float) -> sitk.Image:
    arr = sitk.GetArrayFromImage(image)
    mask_arr = sitk.GetArrayFromImage(mask) > 0
    if not mask_arr.any():
        raise RuntimeError("Empty brain mask -- skull-stripping failed for this subject.")
    lo, hi = np.percentile(arr[mask_arr], [low_pct, high_pct])
    arr = np.clip(arr, lo, hi)
    arr[~mask_arr] = lo
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(image)
    return out


def extract_axial_slices(image: sitk.Image, mask: sitk.Image, min_brain_frac: float):
    arr = sitk.GetArrayFromImage(image)  # z, y, x
    mask_arr = sitk.GetArrayFromImage(mask) > 0
    slice_area = mask_arr.shape[1] * mask_arr.shape[2]
    slices = []
    for z in range(arr.shape[0]):
        frac = mask_arr[z].sum() / slice_area
        if frac >= min_brain_frac:
            slices.append((z, arr[z]))
    return slices


def process_subject(row, args, nifti_dir: Path, slices_dir: Path):
    subject_id, image_id = row["subject_id"], row["image_id"]
    dicom_dir = Path(args.dicom_root) / row["dicom_directory"]
    tag = f"{subject_id}_{image_id}"

    volume = dicom_series_to_nifti(dicom_dir)
    volume = n4_bias_correct(volume)

    nifti_dir.mkdir(parents=True, exist_ok=True)
    volume_path = nifti_dir / f"{tag}.nii.gz"
    sitk.WriteImage(volume, str(volume_path))

    if args.skull_strip_method == "synthstrip":
        brain_path, mask_path = skull_strip_synthstrip(volume_path, nifti_dir / "synthstrip", Path(args.synthstrip_model), use_gpu=args.gpu)
        brain = sitk.ReadImage(str(brain_path))
        mask = sitk.Cast(sitk.ReadImage(str(mask_path)) > 0, sitk.sitkUInt8)
    else:
        mask = skull_strip_otsu(volume)
        brain = sitk.Mask(volume, mask)

    if args.mni_template:
        brain = register_to_template(brain, args.mni_template)
        mask = sitk.Cast(brain != 0, sitk.sitkUInt8)

    brain = resample_isotropic(brain, args.isotropic_spacing, is_mask=False)
    mask = resample_isotropic(mask, args.isotropic_spacing, is_mask=True)

    brain = clip_intensity(brain, mask, args.clip_low_pct, args.clip_high_pct)

    slices_dir.mkdir(parents=True, exist_ok=True)
    out_paths = []
    for z, arr2d in extract_axial_slices(brain, mask, args.min_brain_frac):
        slice_img = sitk.GetImageFromArray(arr2d)
        out_path = slices_dir / f"{tag}_z{z:03d}.nii.gz"
        sitk.WriteImage(slice_img, str(out_path))
        out_paths.append(str(out_path.resolve()))
    return out_paths


def write_manifest(paths, out_path: Path):
    with open(out_path, "w", newline="") as f:
        csv.writer(f, delimiter=",").writerow(paths)
    print(f"Wrote {len(paths)} slice paths to {out_path}")


def main():
    args = parse_args()
    if args.skull_strip_method == "synthstrip":
        ensure_synthstrip_model(Path(args.synthstrip_model))
    participants = pd.read_csv(Path(args.dicom_root) / "participants.csv")
    if args.first_n:
        participants = participants.iloc[: args.first_n]

    output_dir = Path(args.output_dir)
    nifti_dir = output_dir / "nifti"
    slices_dir = output_dir / "slices"

    rng = np.random.RandomState(args.seed)
    subject_ids = participants["subject_id"].unique()
    rng.shuffle(subject_ids)
    n_val = max(1, int(len(subject_ids) * args.val_frac))
    n_test = max(1, int(len(subject_ids) * args.test_frac))
    val_subjects = set(subject_ids[:n_val])
    test_subjects = set(subject_ids[n_val : n_val + n_test])

    split_paths = {"train": [], "val": [], "test": []}
    n_failed = 0
    for _, row in participants.iterrows():
        split = "val" if row["subject_id"] in val_subjects else "test" if row["subject_id"] in test_subjects else "train"
        try:
            paths = process_subject(row, args, nifti_dir, slices_dir)
            split_paths[split].extend(paths)
            print(f"[ok] {row['subject_id']}/{row['image_id']} -> {len(paths)} slices ({split})")
        except Exception as e:
            n_failed += 1
            print(f"[FAIL] {row['subject_id']}/{row['image_id']}: {e}", file=sys.stderr)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(split_paths["train"], output_dir / "train_normal.csv")
    write_manifest(split_paths["val"], output_dir / "val_normal.csv")
    write_manifest(split_paths["test"], output_dir / "test_normal.csv")
    print(f"Done. {n_failed} subject(s) failed.")


if __name__ == "__main__":
    main()
