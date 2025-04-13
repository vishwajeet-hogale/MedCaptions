#!/bin/bash

# Script to train the LSTM-based medical caption model
# Optimized for MacBook with 16GB RAM
# Usage: ./train_lstm_mac.sh [epochs] [batch_size] [learning_rate]

# Default parameters - reduced for Mac
EPOCHS=${1:-20}
BATCH_SIZE=${2:-8}  # Reduced batch size for memory constraints
LEARNING_RATE=${3:-0.0003}
DATASET="med"  # Base name without '_train' suffix
CHECKPOINT_DIR="checkpoints"

echo "=== Training LSTM Medical Caption Model (Mac Optimized) ==="
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE (reduced for Mac memory)"
echo "Learning Rate: $LEARNING_RATE"
echo "Dataset: $DATASET"
echo "==================================="

# Create checkpoint directory if it doesn't exist
mkdir -p $CHECKPOINT_DIR

# Run the training script with LSTM configuration optimized for Mac
python train_medical_caption.py \
  --model_type lstm \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --learning_rate $LEARNING_RATE \
  --lstm_hidden_size 768 \
  --lstm_layers 1 \
  --encoder_lr 0.00001 \
  --checkpoint_dir $CHECKPOINT_DIR \
  --dataset $DATASET \
  --gradient_clip 1.0 \
  --checkpoint_freq 10 \
  --create_new_dataset  # Create the dataset if it doesn't exist

echo "LSTM Training Completed!"
echo "Checkpoints saved in $CHECKPOINT_DIR" 