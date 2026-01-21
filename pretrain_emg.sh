#!/bin/bash

# ============================================================================
# EMG Pretraining Script for PhysioWave
# ============================================================================
# This script demonstrates how to launch distributed pretraining for 
# EMG signals using the BERT Wavelet Transformer architecture.
#
# Usage:
#   1. Modify the paths to point to your EMG data files
#   2. Adjust hyperparameters based on your hardware and dataset
#   3. Run: bash pretrain_emg_example.sh
# ============================================================================

# Number of GPUs to use for distributed training
NUM_GPUS=1

# Data paths (modify these to point to your preprocessed EMG HDF5 files)
# Multiple files can be specified using comma separation
TRAIN_FILES="/lambda/nfs/Kiana/PhysioWave-main9/DB6_processed_8ch/train.h5"
VAL_FILES="/lambda/nfs/Kiana/PhysioWave-main9/DB6_processed_8ch/val.h5"

# Output directory for checkpoints and logs
OUTPUT_DIR="./pretrain_emg_output"

# Create output directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}

# Launch distributed training for EMG
torchrun --nproc_per_node=${NUM_GPUS} pretrain.py \
  --train_files "${TRAIN_FILES}" \
  --val_files "${VAL_FILES}" \
  --in_channels 8 \
  --max_level 3 \
  --wave_kernel_size 16 \
  --wavelet_names sym4 sym5 db6 coif3 bior4.4 \
  --use_separate_channel \
  --patch_size 64 \
  --patch_stride 16 \
  --embed_dim 256 \
  --depth 6 \
  --num_heads 8 \
  --mlp_ratio 4.0 \
  --dropout 0.1 \
  --use_pos_embed \
  --pos_embed_type 2d \
  --max_length 512 \
  --batch_size 32 \
  --num_workers 8 \
  --epochs 30 \
  --lr 1e-4 \
  --weight_decay 0.01 \
  --grad_accumulation_steps 2 \
  --grad_clip 1.0 \
  --use_amp \
  --scheduler cosine \
  --warmup_epochs 10 \
  --mask_ratio 0.6 \
  --masking_strategy frequency_guided \
  --importance_ratio 0.6 \
  --save_freq 10 \
  --seed 42 \
  --output_dir "${OUTPUT_DIR}"

echo "EMG pretraining completed. Results saved to ${OUTPUT_DIR}"
echo "Use the best_model.pth from ${OUTPUT_DIR} for downstream fine-tuning tasks"
