#!/bin/bash
# Baseline: Gaussian noise (simplex_noise=0)
# Step 1: Train VQ-VAE
python train_vqvae.py \
    --output_dir ./outputs \
    --model_name vqvae_brain \
    --training_ids ./data/train_normal.csv \
    --validation_ids ./data/val_normal.csv \
    --spatial_dimension 2 \
    --image_size 128 \
    --vqvae_in_channels 1 \
    --vqvae_out_channels 1 \
    --vqvae_num_embeddings 256 \
    --vqvae_embedding_dim 8 \
    --batch_size 8 \
    --n_epochs 100 \
    --is_grayscale 1

# Step 2: Train DDPM with Gaussian noise (baseline)
python train_ddpm.py \
    --output_dir ./outputs \
    --model_name ddpm_gaussian \
    --training_ids ./data/train_normal.csv \
    --validation_ids ./data/val_normal.csv \
    --vqvae_checkpoint ./outputs/vqvae_brain/checkpoint.pth \
    --spatial_dimension 2 \
    --simplex_noise 0 \
    --model_type small \
    --beta_schedule linear_beta \
    --batch_size 16 \
    --n_epochs 300 \
    --is_grayscale 1

# Step 3: Reconstruct
python reconstruct.py \
    --output_dir ./outputs \
    --model_name ddpm_gaussian \
    --validation_ids ./data/val_normal.csv \
    --in_ids ./data/test_normal.csv \
    --out_ids ./data/test_anomalous.csv \
    --vqvae_checkpoint ./outputs/vqvae_brain/checkpoint.pth \
    --spatial_dimension 2 \
    --simplex_noise 0 \
    --is_grayscale 1

# Step 4: OOD detection score
python ood_detection.py \
    --output_dir ./outputs \
    --model_name ddpm_gaussian
