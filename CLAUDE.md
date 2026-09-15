# SAND — Simplex vs Gaussian Noise LDM for Brain MRI Anomaly Detection

This file gives Claude Code (or any AI assistant) project context on a fresh
checkout, so a new session doesn't have to rediscover it from scratch.

## Research goal

Compare **Simplex vs Gaussian noise** in diffusion-model **unsupervised**
brain MRI anomaly detection. Trained on cognitively normal (CN) T1 scans only;
tested on **Alzheimer's disease (AD)** to see whether the approach applies to
AD.

### Current plan (lab meeting 2026-09-14)

Reproduce first, then change one thing at a time.

1. **Now (conference abstract):** reproduce AnoDDPM (pixel-space DDPM,
   official code) with Gaussian vs Simplex noise on ADNI CN T1 slices through
   the **hippocampus**, evaluated CN vs AD (AUROC). T1 only. Start with all
   hippocampus slices, then trim slices from the range edges as an ablation.
   Region-wise noise / mask-based inpainting (AnoDDPM's own suggested fix) is
   future work.
2. **Later:** Pinaya 2022's latent diffusion model (VQ-VAE + DDPM, KL anomaly
   mask + partial healing) in this repo (`ddpm-ood`), again Gaussian vs
   Simplex.

The AnoDDPM reproduction lives outside this repo: a clone of
`Julian-Wyatt/AnoDDPM` at `~/LimLab/AnoDDPM`, branch `sand-adni` (ADNI
hippocampus data loader, `test_args/args101.json` = Gaussian /
`args102.json` = Simplex, `sand_eval_ad.py` CN-vs-AD evaluation,
`sand_analyze.py` subgroup AUROCs). Run it in its own conda env
(torch 1.13.1, numba, ffmpeg) with `PYTHONNOUSERSITE=1`.

### Key finding

`marksgraham/ddpm-ood` **already has Simplex noise built in** via the
`--simplex_noise 1` flag:

- `src/utils/simplex_noise.py` — `Simplex_CLASS` + `generate_simplex_noise()`
- `src/trainers/base.py` — initializes simplex noise generator if the flag is set
- `src/trainers/ddpm_trainer.py` — uses simplex or gaussian noise in the
  train/val loop

So the core experiment is simply: train with `--simplex_noise 0` (baseline)
vs `--simplex_noise 1` (proposed), then compare anomaly-detection metrics.

Respect the existing `ddpm-ood` framework architecture (MONAI Generative
library, `DiffusionInferer`, `DDPMScheduler`) when making changes — the
simplex/gaussian switch is meant to be a controlled comparison, not a
rewrite.

### Reference papers

Referenced in this project's research, and committed to this repo under
`paper/` (see note below):

1. **Fast Unsupervised Brain Anomaly Detection and Segmentation with
   Diffusion Models** (Pinaya/Graham et al., MICCAI 2022) — the baseline LDM
   framework. Code: `marksgraham/ddpm-ood`.
2. **AnoDDPM: Anomaly Detection with DDPMs using Simplex Noise** (Wyatt et
   al., CVPR 2022 Workshop) — the simplex-noise innovation. Code:
   `Julian-Wyatt/AnoDDPM`.
3. **Generalizable MRI Normative Modelling to Detect Age-Related Changes** —
   third reference paper.

## Repo layout

- `train_vqvae.py`, `train_ddpm.py` — training entry points.
- `reconstruct.py`, `ood_detection.py` — inference / anomaly scoring.
- `preprocess_mri.py` — raw ADNI DICOM (`CN_MRI_raw_486subjects/`, or AD
  data with `--split all_test`) → DICOM→NIfTI → skull-strip (SynthStrip, on
  the raw image) → bias correction (FSL FAST `-B`, default; `--bias_correction
  n4` for the old order) → affine registration to MNI152 1mm
  (`--mni_template`, antspyx) → FreeSurfer SynthSeg on the registered brain
  (`--synthseg`; hippocampus labels 17/53 → per-subject axial z range) →
  0.5/99.5 percentile clip inside the brain, scaled to [0, 1]. Writes
  `volumes/<tag>.npy` (z, y, x) + `subjects.csv` (split, hippocampus range,
  clip values). `--export_slices` additionally writes the 2D slice NIfTIs +
  CSV manifests for `ddpm-ood`. Parallel with `--workers`; SynthStrip queues
  one job per GPU; finished subjects (`volumes/<tag>.json`) are skipped on
  rerun. `--skull_strip_method otsu` is a smoke-test fallback only.
- `third_party/mri_synthstrip.py` — FreeSurfer's official SynthStrip script,
  vendored verbatim (source commit in its header). Weights are fetched from
  MGH into `~/.cache/synthstrip/` and SHA256-checked against the official
  repo's git-annex key.
- `src/` — trainers, networks, losses, data loading (`src/data/`), simplex
  noise utility.
- `configs/train_gaussian.sh`, `configs/train_simplex.sh` — the two
  comparison runs (baseline vs proposed).
- `requirements.txt` — training deps (pins `torch==1.13.1` to match
  `ddpm-ood`). `requirements-preprocess.txt` — preprocessing deps, installed
  in a **separate** environment (on the original machine: conda env
  `sand-preprocess`), because SynthStrip's stack needs a modern torch.
  Always install into / run that env with `PYTHONNOUSERSITE=1` — otherwise
  the conda Python also sees `~/.local` site-packages, and pip will
  uninstall packages the training environment depends on.

## Setup notes / things not obvious from the code

- The three reference PDFs (see above) live in the `paper/` folder and are
  **intentionally committed** to this repo — copyrighted published papers,
  deliberately checked into git since the repo is private.
- `data/`, `checkpoints/`, `runs/`, `outputs/`, `logs/`, model weight files
  (`*.pt`, `*.pth`, `*.ckpt`) are gitignored — machine-specific / too large
  for git. Re-point them per machine.
- No CI/tests yet; verify changes by running the relevant `configs/*.sh`
  script end-to-end on a small subset first.
- Preprocessing tool choices: none of the three reference papers used the
  same pipeline (Pinaya: UK Biobank's pre-processed data; AnoDDPM: NFBS
  *full-skull* images with no skull-stripping or registration, BrainSuite bias
  correction only on its tumour test set; Parker 2025: FreeSurfer 6.0).
  The professor chose SynthStrip + FSL FAST + SynthSeg + MNI152 affine.
  HD-BET was tried first but dropped (its Zenodo-hosted weights were
  unreachable).
- AnoDDPM's own intensity normalisation (clip to mean−1·std … mean+2·std of
  the whole volume) destroys skull-stripped images (it clips 20–70% of brain
  voxels), so the SAND percentile clip is used instead.
- Scanner confound: ~19% of CN scans (all 2005–2016) are 1.5T, but every AD
  scan is 3T. Report CN-vs-AD results both overall and 3T-only.
- External tools on the original machine: FSL FAST in conda env `fsl-fast`,
  FreeSurfer 7.4.1 at `/scratch/users/tjdnjs/tools/freesurfer` (SynthSeg).
