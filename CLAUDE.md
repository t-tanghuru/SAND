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
- `preprocess_mri.py` — turns raw ADNI DICOM (e.g. `CN_MRI_raw_486subjects/`)
  into the 2D axial NIfTI slices + `train/val/test_normal.csv` manifests that
  `src/data/get_train_and_val_dataloader.py` expects. Pipeline: DICOM→NIfTI →
  N4 bias correction → skull-strip (SynthStrip) → optional affine
  registration to an MNI template (`--mni_template`, needs antspyx) →
  isotropic resample → percentile intensity clipping → per-slice NIfTI
  export. Only produces the *normal* splits — anomalous test data needs a
  separate lesion dataset (BRATS/WMH/etc.), not covered by this script.
  `--skull_strip_method otsu` exists only as a smoke-test fallback; never
  use it to generate training data.
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
- Skull-stripping tool choice: none of the three reference papers used the
  same tool (Pinaya: UK Biobank's pre-processed data; AnoDDPM: deliberately
  used NFBS *full-skull* images with no skull-stripping or registration, and
  BrainSuite bias correction only on its tumour test set; normative
  modelling: FreeSurfer 6.0).
  SynthStrip was chosen because it is peer-reviewed, accurate, the official
  FreeSurfer tool, and its weights were obtainable (a possible, optional
  extension to dementia would also stay on the FreeSurfer path). HD-BET was tried first but dropped (its
  Zenodo-hosted weights were unreachable, and it isn't on the FreeSurfer path).
