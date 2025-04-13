#!/bin/bash

# Script to train and compare both LSTM and Transformer models
# Usage: ./compare_models.sh [epochs] [batch_size]

# Default parameters
EPOCHS=${1:-15}
BATCH_SIZE=${2:-16}
DATASET="med_train"
CHECKPOINT_DIR="checkpoints"

echo "=== Training and Comparing Medical Caption Models ==="
echo "Epochs per model: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Dataset: $DATASET"
echo "==================================="

# Create checkpoint directory if it doesn't exist
mkdir -p $CHECKPOINT_DIR

# Train LSTM model first
echo ""
echo "Step 1: Training LSTM model..."
echo ""

python train_medical_caption.py \
  --model_type lstm \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --learning_rate 0.0003 \
  --lstm_hidden_size 1024 \
  --lstm_layers 2 \
  --encoder_lr 0.00001 \
  --checkpoint_dir $CHECKPOINT_DIR \
  --dataset $DATASET \
  --gradient_clip 1.0

# Train Transformer model
echo ""
echo "Step 2: Training Transformer model..."
echo ""

python train_medical_caption.py \
  --model_type transformer \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --learning_rate 0.0001 \
  --transformer_model "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" \
  --encoder_lr 0.00001 \
  --checkpoint_dir $CHECKPOINT_DIR \
  --dataset $DATASET \
  --temperature 0.07 \
  --warmup_epochs 2

# Evaluate and compare both models
echo ""
echo "Step 3: Comparing models performance..."
echo ""

# Run comparison using generate_caption_transformer.py with compare flag
python generate_caption_transformer.py \
  --checkpoint "$CHECKPOINT_DIR/best_lstm.pt" \
  --compare \
  --samples 5 \
  --topk 3

echo ""
echo "Model comparison completed!"
echo "Checkpoints saved in $CHECKPOINT_DIR"
echo "Check the loss curve plots to compare training performance." 