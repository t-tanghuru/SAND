"""Preprocess raw ADNI DICOM T1 volumes.

Pipeline per subject/image (`--bias_correction fast`, the lab-meeting 2026-09-14 setting):
  1. DICOM series -> 3D NIfTI volume (SimpleITK).
  2. Skull-stripping with SynthStrip (Hoopes et al. 2022, NeuroImage), run via
     FreeSurfer's official `mri_synthstrip` script vendored verbatim under
     `third_party/` with the official `synthstrip.1.pt` weights (SHA256
     verified). `--skull_strip_method otsu` is a low-quality smoke-test
     fallback only -- never use it to produce training data.
  3. Bias field correction with FSL FAST (`fast -B`, needs a skull-stripped
     input). `--bias_correction n4` restores the older N4-before-stripping order.
  4. Affine registration to the MNI152 1mm brain template (ANTsPy).
  5. Resample to isotropic 1mm spacing (a no-op after registration).
  6. FreeSurfer SynthSeg on the registered brain -> hippocampus (labels 17/53)
     axial range, recorded per subject in subjects.csv (`--synthseg`).
  7. Robust intensity clipping (0.5-99.5 percentile inside the brain mask),
     saved as a [0, 1]-scaled float32 volume for the AnoDDPM loader.
  8. Optionally (`--export_slices`) extract axial slices whose brain-mask
     coverage exceeds --min_brain_frac as 2D NIfTI files for `ddpm-ood`.

Output layout:
  <output_dir>/nifti/...                                   (intermediate volumes)
  <output_dir>/volumes/<subject_id>_<image_id>.npy          (registered, clipped, [0, 1]; z, y, x)
  <output_dir>/subjects.csv                                 (split + hippocampus z range per subject)
  <output_dir>/slices/<subject_id>_<image_id>_z<idx>.nii.gz (--export_slices only)
  <output_dir>/train_normal.csv, val_normal.csv, test_normal.csv (--export_slices only;
    subject-level split; single row of comma-separated slice paths, matching
    the format `ddpm-ood`'s own get_*_dataset.py scripts write)

This only prepares the *normal* (CN) splits -- test_anomalous.csv needs a
separate lesion dataset (e.g. BRATS/WMH) and is out of scope here.
"""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from multiprocessing import Pool
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
    parser.add_argument("--bias_correction", choices=["fast", "n4"], default="fast")
    parser.add_argument("--fsl_dir", default="/scratch/users/tjdnjs/conda-envs/fsl-fast", help="FSLDIR containing bin/fast.")
    parser.add_argument("--synthseg", action="store_true", help="Run FreeSurfer SynthSeg on the registered brain to find the hippocampus.")
    parser.add_argument("--freesurfer_home", default="/scratch/users/tjdnjs/tools/freesurfer")
    parser.add_argument("--export_slices", action="store_true", help="Also write 2D slice NIfTIs + CSV manifests for ddpm-ood.")
    parser.add_argument("--split", choices=["cn", "all_test"], default="cn", help="cn: subject-level train/val/test split; all_test: every subject goes to the test split (patient data).")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads_per_worker", type=int, default=4)
    parser.add_argument("--n_gpus", type=int, default=1)
    return parser.parse_args()


def dicom_series_to_nifti(dicom_dir: Path) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(dicom_dir))
    if not series_ids:
        raise RuntimeError(f"No DICOM series found in {dicom_dir}")
    files = reader.GetGDCMSeriesFileNames(str(dicom_dir), series_ids[0])
    reader.SetFileNames(files)
    image = reader.Execute()
    # Enhanced (multi-frame) MR DICOM stores the whole volume in one file and is read as 4D with a trailing size-1 axis.
    if image.GetDimension() == 4 and image.GetSize()[3] == 1:
        image = image[:, :, :, 0]
    return image


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


def fast_bias_correct(brain_path: Path, out_dir: Path, fsl_dir: str) -> Path:
    """FSL FAST 3-class T1 segmentation; `-B` writes the bias-corrected brain as <prefix>_restore."""
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / brain_path.name.replace(".nii.gz", "")
    env = dict(os.environ, FSLDIR=fsl_dir, FSLOUTPUTTYPE="NIFTI_GZ")
    cmd = [f"{fsl_dir}/bin/fast", "-t", "1", "-n", "3", "-B", "-o", str(prefix), str(brain_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    restore = Path(f"{prefix}_restore.nii.gz")
    if result.returncode != 0 or not restore.exists():
        raise RuntimeError(f"FSL fast failed: {(result.stderr or result.stdout)[-2000:]}")
    return restore


def synthseg(image_path: Path, out_path: Path, freesurfer_home: str, threads: int) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, FREESURFER_HOME=freesurfer_home)
    cmd = [f"{freesurfer_home}/bin/mri_synthseg", "--i", str(image_path), "--o", str(out_path), "--cpu", "--threads", str(threads)]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0 or not out_path.exists():
        raise RuntimeError(f"mri_synthseg failed: {(result.stderr or result.stdout)[-2000:]}")
    return out_path


def hippocampus_z_range(seg: sitk.Image, reference: sitk.Image):
    """Axial (z) extent and centroid of SynthSeg left/right hippocampus (labels 17, 53) on the reference grid."""
    seg = sitk.Resample(seg, reference, sitk.Transform(), sitk.sitkNearestNeighbor, 0)
    arr = np.isin(sitk.GetArrayFromImage(seg), (17, 53))
    zs = np.nonzero(arr.any(axis=(1, 2)))[0]
    if len(zs) == 0:
        raise RuntimeError("SynthSeg found no hippocampus.")
    per_z = arr.sum(axis=(1, 2))
    return int(zs.min()), int(zs.max()), float((per_z * np.arange(len(per_z))).sum() / per_z.sum()), int(arr.sum())


def skull_strip_synthstrip(nifti_path: Path, out_dir: Path, model_path: Path, use_gpu: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = nifti_path.name.replace(".nii.gz", "")
    brain_path = out_dir / f"{stem}_brain.nii.gz"
    mask_path = out_dir / f"{stem}_mask.nii.gz"
    cmd = [sys.executable, str(SYNTHSTRIP_SCRIPT), "-i", str(nifti_path), "-o", str(brain_path), "-m", str(mask_path), "--model", str(model_path)]
    if use_gpu:
        cmd.append("-g")
    # SynthStrip can take ~13 GB on large volumes, so parallel workers queue for their GPU one at a time.
    lock_path = Path(tempfile.gettempdir()) / f"sand_synthstrip_gpu{os.environ.get('CUDA_VISIBLE_DEVICES', 'cpu')}.lock"
    with open(lock_path, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        result = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ))
    if result.returncode != 0:
        raise RuntimeError(f"mri_synthstrip failed: {(result.stderr or result.stdout)[-2000:]}")
    return brain_path, mask_path


def register_to_template(brain_image: sitk.Image, mask: sitk.Image, template_path: str):
    """Affine-register the brain to the template and carry the brain mask along with the same transform."""
    if not HAS_ANTS:
        raise RuntimeError("antspyx is not installed (`pip install antspyx`) -- required for --mni_template registration.")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        sitk.WriteImage(brain_image, str(tmp / "moving.nii.gz"))
        sitk.WriteImage(mask, str(tmp / "mask.nii.gz"))
        fixed = ants.image_read(template_path)
        result = ants.registration(fixed=fixed, moving=ants.image_read(str(tmp / "moving.nii.gz")), type_of_transform="Affine")
        warped_mask = ants.apply_transforms(
            fixed=fixed,
            moving=ants.image_read(str(tmp / "mask.nii.gz")),
            transformlist=result["fwdtransforms"],
            interpolator="nearestNeighbor",
        )
        ants.image_write(result["warpedmovout"], str(tmp / "brain_reg.nii.gz"))
        ants.image_write(warped_mask, str(tmp / "mask_reg.nii.gz"))
        brain_reg = sitk.ReadImage(str(tmp / "brain_reg.nii.gz"))
        mask_reg = sitk.Cast(sitk.ReadImage(str(tmp / "mask_reg.nii.gz")) > 0, sitk.sitkUInt8)
        mask_reg.CopyInformation(brain_reg)
        return brain_reg, mask_reg


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


def process_subject(row, args, nifti_dir: Path, slices_dir: Path, volumes_dir: Path):
    subject_id, image_id = row["subject_id"], row["image_id"]
    dicom_dir = Path(args.dicom_root) / row["dicom_directory"]
    tag = f"{subject_id}_{image_id}"

    volume = dicom_series_to_nifti(dicom_dir)
    if args.bias_correction == "n4":
        volume = n4_bias_correct(volume)
    else:
        volume = sitk.Cast(volume, sitk.sitkFloat32)

    nifti_dir.mkdir(parents=True, exist_ok=True)
    volume_path = nifti_dir / f"{tag}.nii.gz"
    sitk.WriteImage(volume, str(volume_path))

    if args.skull_strip_method == "synthstrip":
        brain_path, mask_path = skull_strip_synthstrip(volume_path, nifti_dir / "synthstrip", Path(args.synthstrip_model), use_gpu=args.gpu)
        brain = sitk.ReadImage(str(brain_path))
        mask = sitk.Cast(sitk.ReadImage(str(mask_path)) > 0, sitk.sitkUInt8)
        mask.CopyInformation(brain)
    else:
        mask = skull_strip_otsu(volume)
        brain = volume

    # SynthStrip fills background with min(image, 0), which can be a tiny negative from N4 -- zero it explicitly.
    brain = sitk.Mask(brain, mask)

    if args.bias_correction == "fast":
        masked_path = nifti_dir / "synthstrip" / f"{tag}_brain_masked.nii.gz"
        sitk.WriteImage(brain, str(masked_path))
        brain = sitk.ReadImage(str(fast_bias_correct(masked_path, nifti_dir / "fast", args.fsl_dir)))
        brain = sitk.Mask(sitk.Cast(brain, sitk.sitkFloat32), mask)

    if args.mni_template:
        brain, mask = register_to_template(brain, mask, args.mni_template)
        brain = sitk.Mask(brain, mask)

    brain = resample_isotropic(brain, args.isotropic_spacing, is_mask=False)
    mask = resample_isotropic(mask, args.isotropic_spacing, is_mask=True)

    info = {"tag": tag, "subject_id": subject_id, "image_id": image_id}
    if args.synthseg:
        reg_dir = nifti_dir / "registered"
        reg_dir.mkdir(parents=True, exist_ok=True)
        reg_path = reg_dir / f"{tag}_brain_mni.nii.gz"
        sitk.WriteImage(brain, str(reg_path))
        seg_path = synthseg(reg_path, nifti_dir / "synthseg" / f"{tag}_synthseg.nii.gz", args.freesurfer_home, args.threads_per_worker)
        zmin, zmax, zc, nvox = hippocampus_z_range(sitk.ReadImage(str(seg_path)), brain)
        info.update(hippo_zmin=zmin, hippo_zmax=zmax, hippo_zcenter=round(zc, 2), hippo_voxels=nvox)

    arr = sitk.GetArrayFromImage(brain).astype(np.float32)
    mask_arr = sitk.GetArrayFromImage(mask) > 0
    if not mask_arr.any():
        raise RuntimeError("Empty brain mask -- skull-stripping failed for this subject.")
    lo, hi = np.percentile(arr[mask_arr], [args.clip_low_pct, args.clip_high_pct])
    info.update(clip_lo=float(lo), clip_hi=float(hi))
    volumes_dir.mkdir(parents=True, exist_ok=True)
    scaled = (np.clip(arr, lo, hi) - lo) / (hi - lo)
    scaled[~mask_arr] = 0.0
    np.save(volumes_dir / f"{tag}.npy", scaled.astype(np.float32))
    # written last, so an existing .json marks a finished subject that a rerun can skip
    (volumes_dir / f"{tag}.json").write_text(json.dumps(info))

    out_paths = []
    if args.export_slices:
        brain = clip_intensity(brain, mask, args.clip_low_pct, args.clip_high_pct)
        slices_dir.mkdir(parents=True, exist_ok=True)
        for z, arr2d in extract_axial_slices(brain, mask, args.min_brain_frac):
            slice_img = sitk.GetImageFromArray(arr2d)
            out_path = slices_dir / f"{tag}_z{z:03d}.nii.gz"
            sitk.WriteImage(slice_img, str(out_path))
            out_paths.append(str(out_path.resolve()))
    return info, out_paths


def _worker_init(n_gpus: int, threads: int, counter):
    with counter.get_lock():
        idx = counter.value
        counter.value += 1
    # Respect a CUDA_VISIBLE_DEVICES set by the caller (e.g. to keep off GPUs busy with training).
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu_ids = visible.split(",") if visible else [str(i) for i in range(max(n_gpus, 1))]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids[idx % len(gpu_ids)]
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(threads)
    os.environ["OMP_NUM_THREADS"] = str(threads)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(threads)


def _run_one(job):
    row, split, args, nifti_dir, slices_dir, volumes_dir = job
    done = volumes_dir / f"{row['subject_id']}_{row['image_id']}.json"
    if done.exists() and not args.export_slices:
        info = json.loads(done.read_text())
        info["split"] = split
        return info, [], None
    try:
        info, paths = process_subject(row, args, nifti_dir, slices_dir, volumes_dir)
        info["split"] = split
        return info, paths, None
    except Exception as e:
        return {"tag": f"{row['subject_id']}_{row['image_id']}", "subject_id": row["subject_id"], "image_id": row["image_id"], "split": split}, [], str(e)


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

    volumes_dir = output_dir / "volumes"
    split_paths = {"train": [], "val": [], "test": []}
    jobs = []
    for _, row in participants.iterrows():
        if args.split == "all_test":
            split = "test"
        else:
            split = "val" if row["subject_id"] in val_subjects else "test" if row["subject_id"] in test_subjects else "train"
        jobs.append((row.to_dict(), split, args, nifti_dir, slices_dir, volumes_dir))

    import multiprocessing as mp
    counter = mp.Value("i", 0)
    rows, n_failed = [], 0
    with Pool(args.workers, initializer=_worker_init, initargs=(args.n_gpus, args.threads_per_worker, counter)) as pool:
        for info, paths, err in pool.imap_unordered(_run_one, jobs):
            if err is None:
                rows.append(info)
                split_paths[info["split"]].extend(paths)
                hippo = f", hippocampus z {info['hippo_zmin']}-{info['hippo_zmax']}" if "hippo_zmin" in info else ""
                print(f"[ok] {info['subject_id']}/{info['image_id']} ({info['split']}{hippo})", flush=True)
            else:
                n_failed += 1
                print(f"[FAIL] {info['subject_id']}/{info['image_id']}: {err}", file=sys.stderr, flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("tag").to_csv(output_dir / "subjects.csv", index=False)
    if args.export_slices:
        write_manifest(split_paths["train"], output_dir / "train_normal.csv")
        write_manifest(split_paths["val"], output_dir / "val_normal.csv")
        write_manifest(split_paths["test"], output_dir / "test_normal.csv")
    print(f"Done. {n_failed} subject(s) failed.")


if __name__ == "__main__":
    main()
