#!/bin/bash

# Script to train the Transformer-based medical caption model
# Usage: ./train_transformer.sh [epochs] [batch_size] [learning_rate]

# Default parameters
EPOCHS=${1:-30}
BATCH_SIZE=${2:-16}
LEARNING_RATE=${3:-0.0001}
TRANSFORMER_MODEL="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
DATASET="med_train"
CHECKPOINT_DIR="checkpoints"

echo "=== Training Transformer Medical Caption Model ==="
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LEARNING_RATE"
echo "Transformer Model: $TRANSFORMER_MODEL"
echo "Dataset: $DATASET"
echo "==================================="

# Create checkpoint directory if it doesn't exist
mkdir -p $CHECKPOINT_DIR

# Run the training script with Transformer configuration
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
  --warmup_epochs 2

echo "Transformer Training Completed!"
echo "Checkpoints saved in $CHECKPOINT_DIR" 