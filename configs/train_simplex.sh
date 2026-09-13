#!/bin/bash
# Proposed: Simplex noise (simplex_noise=1)
# Assumes VQ-VAE is already trained (reuse from gaussian experiment)

# Train DDPM with Simplex noise (proposed method)
python train_ddpm.py \
    --output_dir ./outputs \
    --model_name ddpm_simplex \
    --training_ids ./data/train_normal.csv \
    --validation_ids ./data/val_normal.csv \
    --vqvae_checkpoint ./outputs/vqvae_brain/checkpoint.pth \
    --spatial_dimension 2 \
    --simplex_noise 1 \
    --model_type small \
    --beta_schedule linear_beta \
    --batch_size 16 \
    --n_epochs 300 \
    --is_grayscale 1

# Reconstruct
python reconstruct.py \
    --output_dir ./outputs \
    --model_name ddpm_simplex \
    --validation_ids ./data/val_normal.csv \
    --in_ids ./data/test_normal.csv \
    --out_ids ./data/test_anomalous.csv \
    --vqvae_checkpoint ./outputs/vqvae_brain/checkpoint.pth \
    --spatial_dimension 2 \
    --simplex_noise 1 \
    --is_grayscale 1

# OOD detection score
python ood_detection.py \
    --output_dir ./outputs \
    --model_name ddpm_simplex
