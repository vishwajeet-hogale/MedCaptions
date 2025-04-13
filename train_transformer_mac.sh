#!/bin/bash

# Script to train the Transformer-based medical caption model
# Optimized for MacBook with 16GB RAM
# Usage: ./train_transformer_mac.sh [epochs] [batch_size] [learning_rate]

# Default parameters - reduced for Mac
EPOCHS=${1:-20}
BATCH_SIZE=${2:-4}  # Significantly reduced for memory constraints
LEARNING_RATE=${3:-0.0001}
# Using a smaller transformer model to reduce memory usage
TRANSFORMER_MODEL="distilbert/distilbert-base-uncased"
DATASET="med"  # Base name without '_train' suffix
CHECKPOINT_DIR="checkpoints"

echo "=== Training Transformer Medical Caption Model (Mac Optimized) ==="
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE (reduced for Mac memory)"
echo "Learning Rate: $LEARNING_RATE"
echo "Transformer Model: $TRANSFORMER_MODEL (smaller model for Mac)"
echo "Dataset: $DATASET"
echo "==================================="

# Create checkpoint directory if it doesn't exist
mkdir -p $CHECKPOINT_DIR

# Run the training script with Transformer configuration optimized for Mac
python train_medical_caption.py \
  --model_type transformer \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --learning_rate $LEARNING_RATE \
  --transformer_model $TRANSFORMER_MODEL \
  --encoder_lr 0.00001 \
  --checkpoint_dir $CHECKPOINT_DIR \
  --dataset $DATASET \
  --temperature 0.07 \
  --warmup_epochs 1 \
  --checkpoint_freq 10 \
  --freeze_encoder \
  --create_new_dataset  # Create the dataset if it doesn't exist

echo "Transformer Training Completed!"
echo "Checkpoints saved in $CHECKPOINT_DIR" 