import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import argparse
import time
import math
from datetime import datetime

# Import both architectures
from caption_lstm import CaptionLSTM 
from transformer_decoder import TransformerMedicalDecoder
from deit_encoder import DeiTMedicalEncoder
from dataloader import get_multicare_dataloader

# ====================
# DEVICE SETUP
# ====================
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ====================
# LOSSES
# ====================
def hybrid_loss(output, target):
    """Combined loss - Cosine + MSE"""
    mse = nn.MSELoss()(output, target)
    cosine = 1 - nn.CosineSimilarity()(output, target).mean()
    return 0.7*cosine + 0.3*mse  # Weighted combination

def compute_info_nce_loss(embeddings_a, embeddings_b, temperature=0.07):
    """
    Compute InfoNCE contrastive loss for matched image-caption pairs
    """
    batch_size = embeddings_a.size(0)
    
    # Normalize embeddings
    embeddings_a = F.normalize(embeddings_a, p=2, dim=1)
    embeddings_b = F.normalize(embeddings_b, p=2, dim=1)
    
    # Compute similarities for all pairs
    sim_matrix = torch.matmul(embeddings_a, embeddings_b.T) / temperature
    
    # Labels: diagonal is positive pairs (matching image-caption)
    labels = torch.arange(batch_size).to(device)
    
    # Computing loss in both directions
    loss_a = F.cross_entropy(sim_matrix, labels)
    loss_b = F.cross_entropy(sim_matrix.T, labels)
    
    # Average bidirectional loss
    return (loss_a + loss_b) / 2.0

# ====================
# BERT MODEL FOR CAPTION EMBEDDING
# ====================
def load_bert_model():
    """Load BERT model for caption embedding"""
    print("Loading BERT model for caption embedding...")
    tokenizer = BertTokenizer.from_pretrained("./MediCareBertTokenizer")
    bert_model = BertModel.from_pretrained("./MediCareBertModel").to(device)
    bert_model.eval()  # BERT is used for inference only
    return tokenizer, bert_model

def get_caption_embedding(caption, tokenizer, bert_model):
    """Generate BERT CLS embedding for a given caption."""
    inputs = tokenizer(caption, return_tensors='pt', truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :]  # CLS token embedding

# ====================
# PLOTTING & REPORTING FUNCTIONS
# ====================
def update_live_plot(current_epoch, batch_losses, epoch_times, batches_per_epoch, final_update=False):
    """Update live plot during training"""
    # Clear the current figure
    plt.clf()

    # Create subplots
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)

    # Plot batch losses
    ax1.plot(batch_losses, label='Batch Loss')
    ax1.set_title(f'Epoch {current_epoch} Batch Losses')
    ax1.set_xlabel('Batch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)

    # Plot epoch trends if available
    if len(epoch_times) > 1:
        ax2.plot(range(1, current_epoch+1), epoch_times, 'bo-')
        ax2.set_title('Training Progress')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Time (s)')
        ax2.grid(True)

    plt.tight_layout()
    plt.pause(0.1)
    
    if final_update:
        plt.savefig('training_progress.png')

def estimate_remaining_time(current_epoch, total_epochs, epoch_times):
    """Estimate remaining training time"""
    if len(epoch_times) < 2:
        return "Calculating..."

    avg_time = sum(epoch_times) / len(epoch_times)
    remaining = avg_time * (total_epochs - current_epoch - 1)

    if remaining > 3600:
        return f'{remaining/3600:.1f} hours'
    elif remaining > 60:
        return f'{remaining/60:.1f} minutes'
    else:
        return f'{remaining:.0f} seconds'

# ====================
# CONFIGURATION
# ====================
def get_args():
    parser = argparse.ArgumentParser(description='Train a medical image captioning model')
    
    # Architecture options
    parser.add_argument('--model_type', type=str, choices=['lstm', 'transformer'], default='transformer',
                        help='Type of decoder model to use (lstm or transformer)')
    
    # Data parameters
    parser.add_argument('--dataset', type=str, default='med_train', help='Dataset name')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--create_new_dataset', action='store_true', help='Create a new dataset')
    
    # Model parameters
    parser.add_argument('--transformer_model', type=str, 
                        default='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
                        help='Transformer model to use (for transformer model type)')
    parser.add_argument('--lstm_hidden_size', type=int, default=1024, 
                        help='Hidden size for LSTM (for lstm model type)')
    parser.add_argument('--lstm_layers', type=int, default=2, 
                        help='Number of LSTM layers (for lstm model type)')
    parser.add_argument('--freeze_encoder', action='store_true', help='Freeze encoder parameters')
    parser.add_argument('--encoder_embed_size', type=int, default=768, help='Encoder embedding size')
    parser.add_argument('--encoder_model', type=str, default='deit_small_patch16_224', 
                        help='Encoder model variant')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--encoder_lr', type=float, default=1e-5, 
                        help='Separate learning rate for encoder (if not frozen)')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=2, help='Warmup epochs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--checkpoint_freq', type=int, default=5, help='Checkpoint frequency (epochs)')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--gradient_clip', type=float, default=1.0, help='Gradient clipping value')
    
    # Loss parameters
    parser.add_argument('--temperature', type=float, default=0.07, help='Temperature for InfoNCE loss')
    
    return parser.parse_args()

# ====================
# TRAINING FUNCTIONS
# ====================
def train_epoch(encoder, decoder, train_loader, optimizer, tokenizer, 
               bert_model, epoch, args):
    """Train for one epoch"""
    encoder.train()
    decoder.train()
    
    total_loss = 0.0
    batch_count = 0
    batch_losses = []
    
    # Create or clear the figure for plotting
    if epoch == 0:
        plt.figure(figsize=(12, 5))
        
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader),
                       desc=f"Epoch {epoch+1}/{args.epochs}", unit='batch')
    
    for batch_idx, batch in progress_bar:
        # Get data
        images = batch['image'].to(device)
        captions = batch['caption']
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Get image features from encoder
        image_features = encoder(images)
        
        # Get caption embeddings from BERT
        with torch.no_grad():
            caption_embeddings = torch.stack([
                get_caption_embedding(c, tokenizer, bert_model).squeeze() 
                for c in captions
            ])
        
        # Forward through decoder - both architectures use the same interface
        dummy_input = torch.zeros((images.size(0), 1, 768)).to(device)  # Placeholder
        pred_caption_embeddings = decoder(dummy_input, image_features)
        
        # Compute appropriate loss based on model type
        if args.model_type == 'lstm':
            loss = hybrid_loss(pred_caption_embeddings, caption_embeddings)
        else:  # transformer
            loss = compute_info_nce_loss(
                pred_caption_embeddings, 
                caption_embeddings,
                temperature=args.temperature
            )
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        nn.utils.clip_grad_norm_(decoder.parameters(), args.gradient_clip)
        nn.utils.clip_grad_norm_(encoder.parameters(), args.gradient_clip)
        
        # Update weights
        optimizer.step()
        
        # Update statistics
        total_loss += loss.item()
        batch_count += 1
        batch_losses.append(loss.item())
        
        # Update progress bar
        progress_bar.set_postfix({"Loss": f"{total_loss/batch_count:.4f}"})
        
    # Final epoch loss
    epoch_loss = total_loss / batch_count
    
    return epoch_loss, batch_losses

def validate(encoder, decoder, val_loader, tokenizer, bert_model, args):
    """Validate the model"""
    encoder.eval()
    decoder.eval()
    
    total_loss = 0.0
    batch_count = 0
    
    progress_bar = tqdm(val_loader, desc="Validation")
    
    with torch.no_grad():
        for batch in progress_bar:
            # Get data
            images = batch['image'].to(device)
            captions = batch['caption']
            
            # Get image features from encoder
            image_features = encoder(images)
            
            # Get caption embeddings from BERT
            caption_embeddings = torch.stack([
                get_caption_embedding(c, tokenizer, bert_model).squeeze() 
                for c in captions
            ])
            
            # Forward through decoder
            dummy_input = torch.zeros((images.size(0), 1, 768)).to(device)
            pred_caption_embeddings = decoder(dummy_input, image_features)
            
            # Compute appropriate loss based on model type
            if args.model_type == 'lstm':
                loss = hybrid_loss(pred_caption_embeddings, caption_embeddings)
            else:  # transformer
                loss = compute_info_nce_loss(
                    pred_caption_embeddings, 
                    caption_embeddings,
                    temperature=args.temperature
                )
            
            # Update statistics
            total_loss += loss.item()
            batch_count += 1
            
            # Update progress bar
            progress_bar.set_postfix({"Val Loss": f"{total_loss/batch_count:.4f}"})
    
    return total_loss / batch_count

def save_checkpoint(encoder, decoder, optimizer, epoch, val_loss, args):
    """Save a checkpoint of the model"""
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Create unique prefix for this run
    model_prefix = f"{args.model_type}"
    
    checkpoint = {
        'epoch': epoch,
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'optimizer': optimizer.state_dict(),
        'val_loss': val_loss,
        'args': args
    }
    
    # Save latest checkpoint
    torch.save(checkpoint, os.path.join(args.checkpoint_dir, f'latest_{model_prefix}.pt'))
    
    # Save epoch checkpoint
    if (epoch + 1) % args.checkpoint_freq == 0:
        torch.save(checkpoint, os.path.join(args.checkpoint_dir, 
                                           f'checkpoint_{model_prefix}_epoch_{epoch+1}.pt'))
    
    # Save best checkpoint
    if not hasattr(save_checkpoint, 'best_val_loss') or val_loss < save_checkpoint.best_val_loss:
        save_checkpoint.best_val_loss = val_loss
        torch.save(checkpoint, os.path.join(args.checkpoint_dir, f'best_{model_prefix}.pt'))
        return True
    return False

# ====================
# SCHEDULERS
# ====================
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Create learning rate scheduler with warmup and cosine decay"""
    def lr_lambda(current_step):
        # Warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return LambdaLR(optimizer, lr_lambda)

# ====================
# MAIN TRAINING LOOP
# ====================
def main():
    # Get arguments
    args = get_args()
    
    # Create data loaders
    print("Creating data loaders...")
    # Define filters for medical images
    caption_filters = [
        {'field': 'label', 'string_list': ['mri', 'head']},
        {'field': 'caption', 'string_list': ['showing', 'demonstrates', 'reveals'], 'operator': 'any'}
    ]
    
    # Ensure dataset names don't get double suffixes
    train_dataset = args.dataset if args.dataset.endswith('_train') else f"{args.dataset}_train"
    val_dataset = args.dataset if args.dataset.endswith('_val') else f"{args.dataset}_val"
    
    # Create train and validation loaders
    train_loader = get_multicare_dataloader(
        dataset_name=train_dataset,
        batch_size=args.batch_size,
        create_new=args.create_new_dataset,
        filters=caption_filters,
        shuffle=True
    )
    
    val_loader = get_multicare_dataloader(
        dataset_name=val_dataset,
        batch_size=args.batch_size,
        create_new=args.create_new_dataset,
        filters=caption_filters,
        shuffle=False
    )
    
    # Load BERT model for caption embedding
    tokenizer, bert_model = load_bert_model()
    
    # Initialize encoder (same for both architectures)
    print("Initializing encoder...")
    encoder = DeiTMedicalEncoder(
        embed_size=args.encoder_embed_size,
        model_name=args.encoder_model,
        pretrained=True
    ).to(device)
    
    # Initialize decoder based on architecture choice
    print(f"Initializing {args.model_type} decoder...")
    if args.model_type == 'lstm':
        decoder = CaptionLSTM(
            hidden_size=args.lstm_hidden_size, 
            num_layers=args.lstm_layers
        ).to(device)
    else:  # transformer
        decoder = TransformerMedicalDecoder(
            model_name=args.transformer_model,
            image_embed_size=384,  # Match DeiT output
            freeze_base=False  # Allow full fine-tuning
        ).to(device)
    
    # Freeze encoder if requested
    if args.freeze_encoder:
        print("Freezing encoder parameters")
        for param in encoder.parameters():
            param.requires_grad = False
    
    # Set up different parameter groups for different learning rates
    encoder_params = list(encoder.parameters())
    decoder_params = list(decoder.parameters())
    
    param_groups = [
        {'params': decoder_params, 'lr': args.learning_rate}
    ]
    
    if not args.freeze_encoder:
        param_groups.append({'params': encoder_params, 'lr': args.encoder_lr})
    
    # Initialize optimizer
    optimizer = optim.AdamW(
        param_groups,
        weight_decay=args.weight_decay
    )
    
    # Calculate scheduler parameters
    num_training_steps = len(train_loader) * args.epochs
    num_warmup_steps = len(train_loader) * args.warmup_epochs
    
    # Learning rate scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            
            # Check if architecture matches
            if 'args' in checkpoint and hasattr(checkpoint['args'], 'model_type'):
                if checkpoint['args'].model_type != args.model_type:
                    print(f"Warning: Checkpoint is from {checkpoint['args'].model_type} model, " 
                          f"but you're training a {args.model_type} model.")
                    if not input("Continue? (y/n): ").lower().startswith('y'):
                        return
            
            # Load model and optimizer state
            encoder.load_state_dict(checkpoint['encoder'])
            
            try:
                decoder.load_state_dict(checkpoint['decoder'])
            except:
                print("Warning: Could not load decoder state. Architecture mismatch. Starting with fresh decoder.")
            
            # Optionally load optimizer state
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
            except:
                print("Warning: Could not load optimizer state. Starting with fresh optimizer.")
            
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
            
            print(f"Resumed from epoch {start_epoch}")
        else:
            print(f"No checkpoint found at {args.resume}")
    
    # Print model info
    print("\n" + "="*50)
    print(f"Training {args.model_type.upper()} model for {args.epochs} epochs")
    print(f"Encoder: {args.encoder_model} (Embedding size: {args.encoder_embed_size})")
    if args.model_type == 'lstm':
        print(f"Decoder: LSTM (Hidden size: {args.lstm_hidden_size}, Layers: {args.lstm_layers})")
    else:
        print(f"Decoder: Transformer (Model: {args.transformer_model})")
    print(f"Batch size: {args.batch_size}, Learning rate: {args.learning_rate}")
    print(f"Encoder learning rate: {args.encoder_lr}")
    print(f"Encoder frozen: {args.freeze_encoder}")
    print("="*50 + "\n")
    
    # Training loop
    print("Starting training...")
    train_losses = []
    val_losses = []
    epoch_times = []
    best_model_found = False
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        epoch_start_time = time.time()
        train_loss, batch_losses = train_epoch(
            encoder, decoder, train_loader, optimizer, 
            tokenizer, bert_model, epoch, args
        )
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(
            encoder, decoder, val_loader, 
            tokenizer, bert_model, args
        )
        val_losses.append(val_loss)
        
        # Update learning rate scheduler
        scheduler.step()
        
        # Epoch timing
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        
        # Save checkpoint
        is_best = save_checkpoint(encoder, decoder, optimizer, epoch, val_loss, args)
        if is_best:
            best_model_found = True
            print(f"New best model found with validation loss: {val_loss:.4f}")
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{args.epochs} completed:")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {epoch_time:.1f}s")
        print(f"Estimated remaining: {estimate_remaining_time(epoch, args.epochs, epoch_times)}")
        
        # Update plot
        update_live_plot(epoch+1, batch_losses, epoch_times, len(train_loader), 
                        final_update=(epoch == args.epochs-1))
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss ({args.model_type.upper()} model)')
    plt.legend()
    plt.savefig(os.path.join(args.checkpoint_dir, f'{args.model_type}_loss_curve.png'))
    
    print("\nTraining completed!")
    if best_model_found:
        print(f"Best model saved with validation loss: {save_checkpoint.best_val_loss:.4f}")
    print(f"Model checkpoints saved in: {args.checkpoint_dir}")

if __name__ == "__main__":
    main() 