# SAND — Simplex vs Gaussian Noise LDM for Brain MRI Anomaly Detection

This file gives Claude Code (or any AI assistant) project context on a fresh
checkout, so a new session doesn't have to rediscover it from scratch.

## Research goal

Replace Gaussian noise with **Simplex noise** in a Latent Diffusion Model
(LDM) for **unsupervised** brain MRI anomaly detection, and demonstrate a
performance/speed advantage.

### Approach (normative modeling)

1. Encode brain MRI → latent space via VQ-VAE.
2. Train a DDPM on **normal** brain MRI latent representations only.
3. At inference: add noise to a patient image up to timestep `t*`, denoise
   back — the model is forced toward a "normal" reconstruction.
4. Residual map = `|original − reconstructed|` → threshold (KL divergence) →
   lesion mask.

### Innovation

- **Baseline** (`ddpm-ood`): Gaussian noise.
- **Proposed**: Simplex noise (spatially coherent, structured, fractal
  patterns), per AnoDDPM.
- **Hypothesis**: Simplex noise better matches brain MRI's spatial structure
  → better anomaly maps.

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

Referenced in this project's research (not committed to this repo — see
note below):

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
- `src/` — trainers, networks, losses, data loading, simplex noise utility.
- `configs/train_gaussian.sh`, `configs/train_simplex.sh` — the two
  comparison runs (baseline vs proposed).
- `requirements.txt` — Python deps.

## Setup notes / things not obvious from the code

- The three reference PDFs (see above) live locally in a `paper/` folder on
  the original machine but are **intentionally not committed** to this repo
  — copyrighted published papers, kept out of git even though the repo is
  private. If you need them on a new machine, transfer them out-of-band
  (zip/cloud/USB), not via this repo.
- `data/`, `checkpoints/`, `runs/`, `outputs/`, `logs/`, model weight files
  (`*.pt`, `*.pth`, `*.ckpt`) are gitignored — machine-specific / too large
  for git. Re-point them per machine.
- No CI/tests yet; verify changes by running the relevant `configs/*.sh`
  script end-to-end on a small subset first.
