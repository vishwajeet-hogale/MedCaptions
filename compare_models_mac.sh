#!/bin/bash

# Script to train and compare both LSTM and Transformer models
# Optimized for MacBook with 16GB RAM
# Usage: ./compare_models_mac.sh [epochs] [batch_size]

# Default parameters - reduced for Mac
EPOCHS=${1:-10}  # Fewer epochs for quicker comparison
BATCH_SIZE=${2:-4}  # Small batch size for memory constraints
DATASET="med"  # Base name without '_train' suffix
CHECKPOINT_DIR="checkpoints"

echo "=== Training and Comparing Medical Caption Models (Mac Optimized) ==="
echo "Epochs per model: $EPOCHS"
echo "Batch Size: $BATCH_SIZE (reduced for Mac memory)"
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
  --lstm_hidden_size 768 \
  --lstm_layers 1 \
  --encoder_lr 0.00001 \
  --checkpoint_dir $CHECKPOINT_DIR \
  --dataset $DATASET \
  --gradient_clip 1.0 \
  --checkpoint_freq 10 \
  --create_new_dataset  # Create the dataset if it doesn't exist

# Train Transformer model
echo ""
echo "Step 2: Training Transformer model..."
echo ""

python train_medical_caption.py \
  --model_type transformer \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --learning_rate 0.0001 \
  --transformer_model "distilbert/distilbert-base-uncased" \
  --encoder_lr 0.00001 \
  --checkpoint_dir $CHECKPOINT_DIR \
  --dataset $DATASET \
  --temperature 0.07 \
  --warmup_epochs 1 \
  --checkpoint_freq 10 \
  --freeze_encoder

# Evaluate and compare both models
echo ""
echo "Step 3: Comparing models performance..."
echo ""

# Run comparison using a smaller sample size for Mac
python generate_caption_transformer.py \
  --checkpoint "$CHECKPOINT_DIR/best_lstm.pt" \
  --compare \
  --samples 3 \
  --topk 3

echo ""
echo "Model comparison completed!"
echo "Checkpoints saved in $CHECKPOINT_DIR"
echo "Check the loss curve plots to compare training performance." 