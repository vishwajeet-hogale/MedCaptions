#!/bin/bash

# Script to train the LSTM-based medical caption model
# Usage: ./train_lstm.sh [epochs] [batch_size] [learning_rate]

# Default parameters
EPOCHS=${1:-30}
BATCH_SIZE=${2:-32}
LEARNING_RATE=${3:-0.0003}
DATASET="med_train"
CHECKPOINT_DIR="checkpoints"

echo "=== Training LSTM Medical Caption Model ==="
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LEARNING_RATE"
echo "Dataset: $DATASET"
echo "==================================="

# Create checkpoint directory if it doesn't exist
mkdir -p $CHECKPOINT_DIR

# Run the training script with LSTM configuration
python train_medical_caption.py \
  --model_type lstm \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --learning_rate $LEARNING_RATE \
  --lstm_hidden_size 1024 \
  --lstm_layers 2 \
  --encoder_lr 0.00001 \
  --checkpoint_dir $CHECKPOINT_DIR \
  --dataset $DATASET \
  --gradient_clip 1.0

echo "LSTM Training Completed!"
echo "Checkpoints saved in $CHECKPOINT_DIR" 