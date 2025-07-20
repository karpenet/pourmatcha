#!/bin/bash

# Vision Transformer Training Script for Task Completion Estimation
# This script trains a ViT using HuggingFace LeRobot datasets

echo "Starting ViT training for task completion estimation with LeRobot datasets..."

# Set environment variables for better performance
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Default dataset IDs - can be overridden with environment variables
POSITIVE_REPOS=${POSITIVE_REPOS:-"user/folding_positive_dataset"}
NEGATIVE_REPOS=${NEGATIVE_REPOS:-"user/folding_negative_dataset"}

echo "Positive datasets: ${POSITIVE_REPOS}"
echo "Negative datasets: ${NEGATIVE_REPOS}"

# Run training with LeRobot datasets
python train_completion_vit.py \
    --positive_repo_ids ${POSITIVE_REPOS} \
    --negative_repo_ids ${NEGATIVE_REPOS} \
    --aggregator sum \
    --batch_size 16 \
    --epochs 10 \
    --lr 5e-6 \
    --dropout 0.3 \
    --image_size 224 \
    --model_name vit_b_16 \
    --val_split 0.1 \
    --seed 42 \
    --precision "bf16" \
    --wandb_project "folding-task-completion-vit-lerobot" \
    --num_frames 1 \
    --save_dir folding_reward_checkpoints \
    --completion_threshold 0.9

echo "Training completed!" 