"""SAND: CN vs AD evaluation of AnoDDPM models trained on ADNI CN hippocampus slices.

For each subject, `n_slices` equally spaced axial slices from the common hippocampus z range (AnoDDPM's evaluation also
used 4 equally spaced slices per volume) are partially diffused to `lambda` and denoised with the official
`forward_backward` (denoise_fn = the model's own noise_fn). The anomaly map is the official squared error
(x - x_hat)^2. Scores per slice: mean squared error inside the brain and inside the SynthSeg hippocampus.
Subject score = mean over its slices. AD has no voxel masks, so the metric is CN-vs-AD AUROC.

Splits: CN val/test from data/v2/subjects.csv; AD (all 'test' in data/v2_AD) is split once with a fixed seed into
a validation part (same size as CN val, used only to choose lambda) and a test part.

usage: python sand_eval_ad.py ARG_NUM --split val --lambdas 100 250 500 [--max_subjects N]
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from sklearn.metrics import roc_auc_score
from torchvision import transforms

from GaussianDiffusion import GaussianDiffusionModel, generate_simplex_noise, get_beta_schedule
from helpers import defaultdict_from_json
from UNet import UNetModel

CN_ROOT = "/home/tjdnjs/LimLab/SAND/data/v2"
AD_ROOT = "/home/tjdnjs/LimLab/SAND/data/v2_AD"
HIPPO_LABELS = (17, 53)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("arg_num")
    p.add_argument("--split", choices=["val", "test"], required=True)
    p.add_argument("--lambdas", type=int, nargs="+", default=[250])
    p.add_argument("--n_slices", type=int, default=4)
    p.add_argument("--max_subjects", type=int, default=None, help="per group, for quick tests")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default="./sand_eval")
    p.add_argument("--ckpt", default="params-final.pt",
                   help="file in model/diff-params-ARGS=N/, e.g. params-epoch1000.pt for the 100k-iteration model")
    return p.parse_args()


def subject_table(split, seed=0):
    cn = pd.read_csv(os.path.join(CN_ROOT, "subjects.csv"))
    train = cn[cn.split == "train"]
    z_range = (int(np.median(train.hippo_zmin)), int(np.median(train.hippo_zmax)))
    cn = cn[cn.split == split].assign(group="CN", label=0, root=CN_ROOT)
    ad = pd.read_csv(os.path.join(AD_ROOT, "subjects.csv")).sort_values("tag").reset_index(drop=True)
    perm = np.random.RandomState(2026).permutation(len(ad))
    n_val = int((pd.read_csv(os.path.join(CN_ROOT, "subjects.csv")).split == "val").sum())
    ad_idx = perm[:n_val] if split == "val" else perm[n_val:]
    ad = ad.iloc[ad_idx].assign(group="AD", label=1, root=AD_ROOT)
    return pd.concat([cn, ad]).reset_index(drop=True), z_range


IMG_T = transforms.Compose([transforms.ToPILImage(), transforms.CenterCrop(235),
                            transforms.Resize((256, 256), transforms.InterpolationMode.BILINEAR),
                            transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
MASK_T = transforms.Compose([transforms.ToPILImage(), transforms.CenterCrop(235),
                             transforms.Resize((256, 256), transforms.InterpolationMode.NEAREST),
                             transforms.ToTensor()])


def pad256(a):
    out = np.zeros((256, 256), dtype=np.float32)
    y0, x0 = (256 - a.shape[0]) // 2, (256 - a.shape[1]) // 2
    out[y0:y0 + a.shape[0], x0:x0 + a.shape[1]] = a
    return out


def load_subject(row, zs):
    vol = np.load(os.path.join(row.root, "volumes", f"{row.tag}.npy"), mmap_mode="r")
    reg = sitk.ReadImage(os.path.join(row.root, "nifti", "registered", f"{row.tag}_brain_mni.nii.gz"))
    seg = sitk.ReadImage(os.path.join(row.root, "nifti", "synthseg", f"{row.tag}_synthseg.nii.gz"))
    seg = sitk.GetArrayFromImage(sitk.Resample(seg, reg, sitk.Transform(), sitk.sitkNearestNeighbor, 0))
    imgs, brain, hippo = [], [], []
    for z in zs:
        s = np.ascontiguousarray(vol[z][::-1]).astype(np.float32)  # same orientation as the training loader
        imgs.append(IMG_T(pad256(s)))
        brain.append(MASK_T(pad256((s > 0).astype(np.float32))) > 0.5)
        hippo.append(MASK_T(pad256(np.isin(seg[z][::-1], HIPPO_LABELS).astype(np.float32))) > 0.5)
    return torch.stack(imgs), torch.stack(brain), torch.stack(hippo)


def main():
    a = parse_args()
    with open(f"test_args/args{a.arg_num}.json") as f:
        args = defaultdict_from_json(json.load(f))
    args["arg_num"] = a.arg_num
    device = torch.device("cuda")
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    ckpt = torch.load(f"./model/diff-params-ARGS={a.arg_num}/{a.ckpt}", map_location="cpu")
    model = UNetModel(args['img_size'][0], args['base_channels'], channel_mults=args['channel_mults'],
                      dropout=args["dropout"], n_heads=args["num_heads"], n_head_channels=args["num_head_channels"],
                      in_channels=1)
    model.load_state_dict(ckpt["ema"])  # official detection code evaluates the EMA weights
    model.to(device).eval()
    del ckpt
    diff = GaussianDiffusionModel(args['img_size'], get_beta_schedule(args['T'], args['beta_schedule']),
                                  loss_weight=args['loss_weight'], loss_type=args['loss-type'], noise=args["noise_fn"],
                                  img_channels=1)
    if args["noise_fn"] == "simplex":
        # the official simplex generator only runs with batch size 1 (it was trained/evaluated that way) but is written
        # to repeat one noise field over the batch; generate it for one image and repeat, which is that intended result
        diff.noise_fn = lambda x, t: generate_simplex_noise(diff.simplex, x[:1], t[:1], False, in_channels=1).repeat(
                x.shape[0], 1, 1, 1)

    subjects, z_range = subject_table(a.split)
    if a.max_subjects:
        subjects = pd.concat([g.head(a.max_subjects) for _, g in subjects.groupby("group")]).reset_index(drop=True)
    zs = np.linspace(z_range[0], z_range[1], a.n_slices + 2)[1:-1].round().astype(int)  # equally spaced, inside the range
    os.makedirs(a.out_dir, exist_ok=True)
    tag = f"args{a.arg_num}_{a.split}" + ("" if a.ckpt == "params-final.pt" else f"_{a.ckpt.replace('.pt', '')}")
    print(f"{tag}: {args['noise_fn']} | CN {int((subjects.group == 'CN').sum())} AD {int((subjects.group == 'AD').sum())} "
          f"| slices z={zs.tolist()} | lambdas {a.lambdas}", flush=True)

    rows = []
    t0 = time.time()
    for i, row in subjects.iterrows():
        x, brain, hippo = load_subject(row, zs)
        x = x.to(device)
        for lam in a.lambdas:
            with torch.no_grad():
                # official detection passes denoise_fn=args["noise_fn"]; for "simplex" that reaches
                # generate_simplex_noise with the same default parameters as noise_fn, so route it through the
                # batch-safe noise_fn above ("gauss" is unchanged)
                x_hat = diff.forward_backward(model, x, see_whole_sequence=None, t_distance=lam,
                                              denoise_fn="noise_fn" if args["noise_fn"] == "simplex" else "gauss")
            sq = (x - x_hat).square().cpu()
            for k, z in enumerate(zs):
                rows.append(dict(tag=row.tag, group=row.group, label=row.label, z=int(z), lam=lam,
                                 brain_mse=float(sq[k][brain[k]].mean()), hippo_mse=float(sq[k][hippo[k]].mean())
                                 if hippo[k].any() else np.nan))
        if i % 10 == 0:
            el = time.time() - t0
            print(f"  {i + 1}/{len(subjects)} subjects, {el / 60:.1f} min elapsed, "
                  f"~{el / (i + 1) * (len(subjects) - i - 1) / 60:.0f} min left", flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(a.out_dir, f"{tag}_slices.csv"), index=False)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out_dir, f"{tag}_slices.csv"), index=False)
    subj = df.groupby(["tag", "group", "label", "lam"], as_index=False)[["brain_mse", "hippo_mse"]].mean()
    subj.to_csv(os.path.join(a.out_dir, f"{tag}_subjects.csv"), index=False)
    summary = []
    for lam, g in subj.groupby("lam"):
        s = df[df.lam == lam]
        summary.append(dict(lam=lam,
                            subject_auroc_brain=roc_auc_score(g.label, g.brain_mse),
                            subject_auroc_hippo=roc_auc_score(g.label, g.hippo_mse),
                            slice_auroc_brain=roc_auc_score(s.label, s.brain_mse),
                            slice_auroc_hippo=roc_auc_score(s.dropna().label, s.dropna().hippo_mse)))
    summary = pd.DataFrame(summary)
    summary.to_csv(os.path.join(a.out_dir, f"{tag}_summary.csv"), index=False)
    print(summary.round(4).to_string(index=False), flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
