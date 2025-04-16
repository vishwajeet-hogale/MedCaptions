import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import argparse
from datetime import datetime

# Local imports
from transformer_decoder import TransformerMedicalDecoder
from deit_encoder import DeiTMedicalEncoder
from dataloader import get_multicare_dataloader

# ====================
# DEVICE SETUP
# ====================
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ====================
# CONFIGURATION
# ====================
def get_args():
    parser = argparse.ArgumentParser(description='Train a transformer-based medical image captioning model')
    
    # Data parameters
    parser.add_argument('--dataset', type=str, default='medCapAll', help='Dataset name')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--create_new_dataset', action='store_true', help='Create a new dataset')
    
    # Model parameters
    parser.add_argument('--transformer_model', type=str, 
                        default='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
                        help='Transformer model to use')
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
    parser.add_argument('--warmup_epochs', type=int, default=0, help='Warmup epochs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--checkpoint_freq', type=int, default=2, help='Checkpoint frequency (epochs)')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    
    # Loss parameters
    parser.add_argument('--margin', type=float, default=0.2, help='Margin for contrastive loss')
    parser.add_argument('--temperature', type=float, default=0.07, help='Temperature for InfoNCE loss')
    
    return parser.parse_args()

# ====================
# BERT MODEL FOR CAPTION EMBEDDING
# ====================
# def load_bert_model():
#     print("Loading BERT model for caption embedding...")
#     tokenizer = BertTokenizer.from_pretrained("./MediCareBertTokenizer")
#     bert_model = BertModel.from_pretrained("./MediCareBertModel").to(device)
#     bert_model.eval()  # BERT is used for inference only
#     return tokenizer, bert_model

def load_bert_model():
    print("Loading BERT model for caption embedding...")
    tokenizer = BertTokenizer.from_pretrained("./MediCareBert")
    bert_model = BertModel.from_pretrained("./MediCareBert").to(device)
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
# TRAINING FUNCTIONS
# ====================
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
    
    # Computing loss in both directions (image→caption and caption→image)
    loss_a = F.cross_entropy(sim_matrix, labels)
    loss_b = F.cross_entropy(sim_matrix.T, labels)
    
    # Average bidirectional loss
    return (loss_a + loss_b) / 2.0

def train_epoch(encoder, decoder, train_loader, optimizer, tokenizer, 
                bert_model, epoch, args):
    """Train for one epoch"""
    encoder.train()
    decoder.train()
    
    total_loss = 0.0
    batch_count = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
    
    for batch in progress_bar:
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
        
        # Forward through decoder (transformer)
        dummy_input = torch.zeros((images.size(0), 1, 768)).to(device)  # Placeholder
        pred_caption_embeddings = decoder(dummy_input, image_features)
        
        # Compute loss
        loss = compute_info_nce_loss(
            pred_caption_embeddings, 
            caption_embeddings,
            temperature=args.temperature
        )
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Update statistics
        total_loss += loss.item()
        batch_count += 1
        
        # Update progress bar
        progress_bar.set_postfix({"Loss": f"{total_loss/batch_count:.4f}"})
    
    return total_loss / batch_count

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
            
            # Forward through decoder (transformer)
            dummy_input = torch.zeros((images.size(0), 1, 768)).to(device)
            pred_caption_embeddings = decoder(dummy_input, image_features)
            
            # Compute loss
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
    
    checkpoint = {
        'epoch': epoch,
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'optimizer': optimizer.state_dict(),
        'val_loss': val_loss,
        'args': args
    }
    
    # Save latest checkpoint
    torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'latest_transformer.pt'))
    
    # Save epoch checkpoint
    if (epoch + 1) % args.checkpoint_freq == 0:
        torch.save(checkpoint, os.path.join(args.checkpoint_dir, f'checkpoint_transformer_epoch_{epoch+1}.pt'))
    
    # Save best checkpoint
    if not hasattr(save_checkpoint, 'best_val_loss') or val_loss < save_checkpoint.best_val_loss:
        save_checkpoint.best_val_loss = val_loss
        torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'best_transformer.pt'))

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
        {'field': 'label', 'string_list': ['radiology']},
        {'field': 'caption', 'string_list': [], 'operator': 'any'}
    ]
    
    # Create train and validation loaders
    train_loader = get_multicare_dataloader(
        dataset_name=f"{args.dataset}_train",
        batch_size=args.batch_size,
        create_new=args.create_new_dataset,
        filters=caption_filters,
        shuffle=True
    )
    
    val_loader = get_multicare_dataloader(
        dataset_name=f"{args.dataset}_val",
        batch_size=args.batch_size,
        create_new=args.create_new_dataset,
        filters=caption_filters,
        shuffle=False
    )
    
    # Load BERT model for caption embedding
    tokenizer, bert_model = load_bert_model()
    
    # Initialize models
    print("Initializing models...")
    encoder = DeiTMedicalEncoder(
        embed_size=args.encoder_embed_size,
        model_name=args.encoder_model,
        pretrained=True
    ).to(device)
    
    decoder = TransformerMedicalDecoder(
        model_name=args.transformer_model,
        image_embed_size=args.encoder_embed_size,  # Match encoder embed size (768)
        freeze_base=False  # Allow full fine-tuning
    ).to(device)
    
    # Freeze encoder if requested
    if args.freeze_encoder:
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
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3,
        verbose=True
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            
            # Load model and optimizer state
            encoder.load_state_dict(checkpoint['encoder'])
            decoder.load_state_dict(checkpoint['decoder'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
            
            print(f"Resumed from epoch {start_epoch}")
        else:
            print(f"No checkpoint found at {args.resume}")
    
    # Training loop
    print("Starting training...")
    train_losses = []
    val_losses = []
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss = train_epoch(
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
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        
        # Save checkpoint
        save_checkpoint(encoder, decoder, optimizer, epoch, val_loss, args)
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(args.checkpoint_dir, 'loss_curve.png'))
    
    print("Training completed!")

if __name__ == "__main__":
    main() 