#!/bin/bash

# ============================================================================
# Fine-tuning Script for BERT Wavelet Transformer
# ============================================================================
# This script demonstrates how to fine-tune a pretrained BERT Wavelet 
# Transformer model for downstream classification tasks.
#
# Usage:
#   1. Modify the paths to point to your dataset files
#   2. Update the pretrained model path
#   3. Adjust num_classes based on your classification task
#   4. Run: bash finetune_example.sh
# ============================================================================

# Number of GPUs to use for distributed training
NUM_GPUS=1

# Data paths (modify these to point to your dataset)
TRAIN_FILE="/lambda/nfs/Kiana/PhysioWave/EPN612_processed/epn612_train_set.h5"
VAL_FILE="/lambda/nfs/Kiana/PhysioWave/EPN612_processed/epn612_val_set.h5"
TEST_FILE="/lambda/nfs/Kiana/PhysioWave/EPN612_processed/epn612_test_set.h5"

# Pretrained model checkpoint
PRETRAINED_MODEL="/lambda/nfs/Kiana/PhysioWave/pretrain_emg_output/best_model.pth"

# Output directory for fine-tuning results
OUTPUT_DIR="./finetune_output"

# Create output directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}

# Launch distributed fine-tuning
torchrun --nproc_per_node=${NUM_GPUS} finetune.py \
  --train_file "${TRAIN_FILE}" \
  --val_file "${VAL_FILE}" \
  --test_file "${TEST_FILE}" \
  --pretrained_path "${PRETRAINED_MODEL}" \
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
  --batch_size 32 \
  --max_length 512 \
  --epochs 5 \
  --lr 2e-4 \
  --weight_decay 1e-3 \
  --grad_clip 1.0 \
  --use_amp \
  --num_workers 8 \
  --world_size ${NUM_GPUS} \
  --scheduler cosine \
  --warmup_epochs 2 \
  --num_classes 6 \
  --pooling mean \
  --head_dropout 0.1 \
  --head_hidden_dim 512 \
  --label_smoothing 0.1 \
  --seed 42 \
  --output_dir "${OUTPUT_DIR}"

echo "Fine-tuning completed. Results saved to ${OUTPUT_DIR}"
echo "Best model checkpoint: ${OUTPUT_DIR}/best_model.pth"
echo "Test results: ${OUTPUT_DIR}/test_results.json"