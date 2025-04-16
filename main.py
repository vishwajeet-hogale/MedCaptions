import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, AutoModel, AutoTokenizer
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms, models
from PIL import Image
import json
import os
import random
from datetime import datetime
from timm import create_model
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import nltk
from nltk.translate.meteor_score import meteor_score
try:
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('punkt')

# ====================
# DEVICE SETUP
# ====================
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ====================
# DEFAULT CONFIGURATION
# ====================
args = {
    'dataset': 'medCapAll2',
    'batch_size': 32,  # Further increased batch size
    'create_new_dataset': True,
    'transformer_model': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
    'freeze_encoder': False,  # Fine-tune everything
    'encoder_embed_size': 512,  # Reduced embedding size for better convergence
    'encoder_model': 'resnet50',  # Switched to ResNet50 instead of DeiT
    'epochs': 30,  # Increased epochs
    'learning_rate': 1e-4,  
    'encoder_lr': 1e-5,
    'weight_decay': 1e-4,
    'warmup_epochs': 5,  # Extended warmup
    'checkpoint_dir': 'checkpoints',
    'checkpoint_freq': 2,
    'resume': None,
    'margin': 0.2,
    'temperature': 0.1
}

# ====================
# DEBUGGING HELPER
# ====================
def debug_tensor(tensor, name="tensor"):
    """Print debug information about a tensor."""
    if tensor is None:
        print(f"{name} is None")
        return
        
    print(f"{name} - shape: {tensor.shape}, dtype: {tensor.dtype}")
    print(f"{name} - min: {tensor.min().item()}, max: {tensor.max().item()}, mean: {tensor.mean().item()}")
    print(f"{name} - has NaN: {torch.isnan(tensor).any()}, has Inf: {torch.isinf(tensor).any()}")

# ====================
# Encoder Model - NEW IMPLEMENTATION
# ====================
class ResNetEncoder(nn.Module):
    def __init__(self, embed_size, pretrained=True):
        super(ResNetEncoder, self).__init__()
        
        # Load pretrained ResNet but remove the final classification layer
        resnet = models.resnet50(pretrained=pretrained)
        modules = list(resnet.children())[:-1]  # Remove the last FC layer
        self.resnet = nn.Sequential(*modules)
        
        # Add a new projection layer
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, embed_size)
        )
        
    def forward(self, x):
        features = self.resnet(x)
        return self.projection(features)

# ====================
# Decoder Model - NEW IMPLEMENTATION
# ====================
class ProjectionDecoder(nn.Module):
    def __init__(self, model_name, image_embed_size, text_embed_size=768):
        super(ProjectionDecoder, self).__init__()
        
        # Load the transformer model for text processing
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # Freeze the transformer (we'll only use it to get text features)
        for param in self.transformer.parameters():
            param.requires_grad = False
            
        # Projection layer for image features
        self.img_projection = nn.Sequential(
            nn.Linear(image_embed_size, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, text_embed_size)
        )
        
    def encode_text(self, text, tokenizer, max_length=128):
        """Encode text using the transformer model."""
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length, padding='max_length')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.transformer(**inputs)
            
        # Get the CLS token embedding
        text_embedding = outputs.last_hidden_state[:, 0, :]
        return text_embedding
        
    def forward(self, image_features, text=None):
        # Project image features to text embedding space
        projected_img_features = self.img_projection(image_features)
        
        # If text is provided, get its embedding for training
        if text is not None:
            text_features = torch.stack([
                self.encode_text(t, self.tokenizer).squeeze() for t in text
            ])
            return projected_img_features, text_features
            
        return projected_img_features

# ====================
# DATASET - IMPROVED
# ====================
class MedicalImageCaptionDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data = []
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Make sure data path directory exists
        os.makedirs("data", exist_ok=True)
        
        # This is a placeholder - in a real implementation,
        # you would load your dataset here
        print(f"Loading dataset from {data_path}")
        
        # For demonstration, create dummy data with more diverse captions
        medical_conditions = [
            "pneumonia", "fracture", "cardiomegaly", "effusion", "pneumothorax",
            "consolidation", "edema", "emphysema", "fibrosis", "atelectasis",
            "infiltration", "mass", "nodule", "pleural thickening"
        ]
        
        medical_sentence_templates = [
            "This medical image shows {} with moderate severity.",
            "Patient presents with {} as seen in the radiograph.",
            "The scan reveals evidence of {} in the lung fields.",
            "The findings are consistent with {}.",
            "Medical imaging demonstrates {} requiring clinical correlation.",
            "This study shows features typical of {}.",
            "The X-ray shows {} in the {} region.",
            "There is evidence of {} on this radiograph.",
            "Imaging reveals {} that needs further evaluation.",
            "This scan demonstrates findings suggestive of {}."
        ]
        
        regions = ["right upper", "right lower", "left upper", "left lower", "bilateral", "perihilar", "peripheral"]
        
        for i in range(10000):  # 100 dummy samples
            condition = random.choice(medical_conditions)
            template = random.choice(medical_sentence_templates)
            
            if "{}" in template and not "{}" in template[template.index("{}")+2:]:
                caption = template.format(condition)
            else:
                region = random.choice(regions)
                caption = template.format(condition, region)
                
            self.data.append({
                'image_path': f"dummy_image_{i}.jpg",
                'caption': caption,
                'label': 'radiology'
            })
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # In a real implementation, you would load the actual image
        # For demonstration, create a dummy image with more structured patterns
        image = torch.rand(3, 224, 224)  # Random tensor representing an image
        
        # Add some structure to the "image" to make it less totally random
        # Use the full idx as seed to ensure all images are different
        random.seed(idx)
        
        # Create a more structured pattern
        for c in range(3):
            # Add a circular pattern
            cx, cy = random.randint(50, 170), random.randint(50, 170)
            radius = random.randint(20, 60)
            for i in range(224):
                for j in range(224):
                    dist = ((i - cx) ** 2 + (j - cy) ** 2) ** 0.5
                    if dist < radius:
                        image[c, i, j] = 0.7 + 0.3 * random.random()
            
            # Add a rectangular pattern for more diversity
            x1, y1 = random.randint(20, 100), random.randint(20, 100)
            x2, y2 = random.randint(120, 200), random.randint(120, 200)
            for i in range(x1, x2):
                for j in range(y1, y2):
                    if i < 224 and j < 224:
                        image[c, i, j] = 0.2 + 0.3 * random.random()
            
            # Add a diagonal pattern
            thickness = random.randint(5, 15)
            start = random.randint(0, 100)
            for i in range(start, 224-start):
                for j in range(max(0, i-thickness), min(224, i+thickness)):
                    image[c, i, j] = 0.5 + 0.5 * random.random()
        
        # Normalize the image
        image = (image - image.mean()) / (image.std() + 1e-5)
        
        return {
            'image': image,
            'caption': item['caption'],
            'label': item['label']
        }

# ====================
# DATA LOADER - SIMPLIFIED
# ====================
def get_multicare_dataloader(dataset_name, batch_size, create_new=True, filters=None, shuffle=True):
    print(f"Creating dataloader for dataset: {dataset_name} (batch size {batch_size})")
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Create dataset
    dataset = MedicalImageCaptionDataset(f"data/{dataset_name}")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Reduced for debugging
        pin_memory=True
    )
    
    return dataloader

# ====================
# LOSS FUNCTION - REVISED
# ====================
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.2, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.temperature = temperature
        
    def forward(self, img_features, text_features):
        # Normalize features
        img_features = F.normalize(img_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)
        
        # Compute similarity matrix
        similarity = torch.matmul(img_features, text_features.t()) / self.temperature
        
        # Labels: diagonal elements (matching pairs) should have higher similarity
        labels = torch.arange(img_features.size(0)).to(similarity.device)
        
        # Compute loss: cross-entropy between similarity and labels
        loss_i2t = F.cross_entropy(similarity, labels)
        loss_t2i = F.cross_entropy(similarity.t(), labels)
        
        # Average bidirectional loss
        loss = (loss_i2t + loss_t2i) / 2.0
        
        return loss

# ====================
# TRAINING - REDESIGNED
# ====================
def train_epoch(encoder, decoder, train_loader, optimizer, epoch, loss_fn, scheduler=None):
    encoder.train()
    decoder.train()
    
    total_loss = 0.0
    batch_count = 0
    
    # Use tqdm for progress bar
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args['epochs']}")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Get data
        images = batch['image'].to(device)
        captions = batch['caption']
        
        # Forward pass through encoder and decoder
        try:
            # Clear gradients
            optimizer.zero_grad()
            
            # Get image embeddings from encoder
            img_features = encoder(images)
            
            # Debug information
            if batch_idx == 0 and epoch == 0:
                debug_tensor(img_features, "img_features")
                
            # Get image and text embeddings from decoder
            img_embeddings, text_embeddings = decoder(img_features, captions)
            
            if batch_idx == 0 and epoch == 0:
                debug_tensor(img_embeddings, "img_embeddings")
                debug_tensor(text_embeddings, "text_embeddings")
                
            # Compute loss
            loss = loss_fn(img_embeddings, text_embeddings)
            
            # Check for NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN or Inf loss detected in batch {batch_idx}, skipping")
                continue
                
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=5.0)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=5.0)
            
            # Update weights
            optimizer.step()
            
            # Update learning rate if scheduler is provided
            if scheduler is not None:
                scheduler.step()
                
            # Update statistics
            total_loss += loss.item()
            batch_count += 1
            
            # Update progress bar
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        except Exception as e:
            print(f"Error in batch {batch_idx}: {str(e)}")
            continue
    
    # Calculate average loss
    avg_loss = total_loss / max(batch_count, 1)
    return avg_loss

# ====================
# VALIDATION - REDESIGNED
# ====================
def validate(encoder, decoder, val_loader, loss_fn):
    encoder.eval()
    decoder.eval()
    
    total_loss = 0.0
    batch_count = 0
    
    # Use tqdm for progress bar
    progress_bar = tqdm(val_loader, desc="Validation")
    
    with torch.no_grad():
        for batch in progress_bar:
            try:
                # Get data
                images = batch['image'].to(device)
                captions = batch['caption']
                
                # Forward pass
                img_features = encoder(images)
                img_embeddings, text_embeddings = decoder(img_features, captions)
                
                # Compute loss
                loss = loss_fn(img_embeddings, text_embeddings)
                
                # Update statistics
                total_loss += loss.item()
                batch_count += 1
                
                # Update progress bar
                progress_bar.set_postfix({"Val Loss": f"{loss.item():.4f}"})
                
            except Exception as e:
                print(f"Error in validation: {str(e)}")
                continue
    
    # Calculate average loss
    avg_loss = total_loss / max(batch_count, 1)
    return avg_loss

# ====================
# INFERENCE AND VISUALIZATION - UPDATED
# ====================
def run_inference_and_visualize(encoder, decoder, train_loader, tokenizer=None, bert_model=None):
    """Run inference on a random batch and visualize results with metrics."""
    encoder.eval()
    decoder.eval()
    
    # Get a random batch from the train loader
    dataloader_iter = iter(train_loader)
    batch = next(dataloader_iter)
    
    images = batch['image'].to(device)
    ground_truth_captions = batch['caption']
    
    # Run inference
    with torch.no_grad():
        # Get image features and projected embeddings
        image_features = encoder(images)
        img_embeddings = decoder(image_features)
        
        # Normalize embeddings
        img_embeddings = F.normalize(img_embeddings, p=2, dim=1)
        
        # Calculate text embeddings for all captions in the batch
        text_embeddings = []
        for caption in ground_truth_captions:
            text_emb = decoder.encode_text(caption, decoder.tokenizer)
            text_embeddings.append(F.normalize(text_emb.squeeze(), p=2, dim=0))
        
        text_embeddings = torch.stack(text_embeddings)
        
        # Calculate similarity matrix
        similarity_matrix = torch.matmul(img_embeddings, text_embeddings.T)
        
        # To prevent always selecting the same caption, we'll use a more diverse 
        # approach to caption selection:
        
        # Option 1: Get the top-k most similar captions and randomly select one
        k = min(3, similarity_matrix.size(1))
        topk_similarities, topk_indices = torch.topk(similarity_matrix, k=k, dim=1)
        
        # Randomly select from top-k for each image
        predicted_indices = []
        for i in range(len(images)):
            selected_idx = topk_indices[i][random.randint(0, k-1)].item()
            predicted_indices.append(selected_idx)
        
        # Option 2: If Option 1 still gives duplicates, we can force unique captions
        # by using a greedy algorithm (uncomment if needed)
        """
        used_caption_indices = set()
        predicted_indices = []
        
        # Sort by highest similarity first
        image_order = torch.argsort(torch.max(similarity_matrix, dim=1)[0], descending=True)
        
        for img_idx in image_order:
            # Get caption similarities for this image
            similarities = similarity_matrix[img_idx]
            
            # Sort caption indices by decreasing similarity
            sorted_caption_indices = torch.argsort(similarities, descending=True)
            
            # Find the most similar caption that hasn't been used yet
            for caption_idx in sorted_caption_indices:
                if caption_idx.item() not in used_caption_indices:
                    predicted_indices.append(caption_idx.item())
                    used_caption_indices.add(caption_idx.item())
                    break
            else:
                # If all captions have been used, just use the most similar one
                predicted_indices.append(sorted_caption_indices[0].item())
        
        # Reorder back to match the original image order
        reordered_indices = [0] * len(image_order)
        for i, img_idx in enumerate(image_order):
            reordered_indices[img_idx] = predicted_indices[i]
        predicted_indices = reordered_indices
        """
        
        # Get the predicted captions
        predicted_captions = [ground_truth_captions[idx] for idx in predicted_indices]
    
    # Calculate metrics (simplified)
    metrics = []
    for i in range(len(ground_truth_captions)):
        try:
            bleu = calculate_bleu(ground_truth_captions[i], predicted_captions[i])
            rouge = calculate_rouge(ground_truth_captions[i], predicted_captions[i])
            meteor = calculate_meteor(ground_truth_captions[i], predicted_captions[i])
            
            metrics.append({
                'bleu': bleu,
                'rouge1': rouge['rouge1'],
                'rouge2': rouge['rouge2'],
                'rougeL': rouge['rougeL'],
                'meteor': meteor
            })
        except Exception as e:
            print(f"Error calculating metrics for sample {i}: {str(e)}")
            metrics.append({
                'bleu': 0.0,
                'rouge1': 0.0,
                'rouge2': 0.0,
                'rougeL': 0.0,
                'meteor': 0.0
            })
    
    # Calculate average metrics
    avg_metrics = {
        'bleu': np.mean([m['bleu'] for m in metrics]),
        'rouge1': np.mean([m['rouge1'] for m in metrics]),
        'rouge2': np.mean([m['rouge2'] for m in metrics]),
        'rougeL': np.mean([m['rougeL'] for m in metrics]),
        'meteor': np.mean([m['meteor'] for m in metrics])
    }
    
    print("\nAverage Metrics:")
    print(f"BLEU: {avg_metrics['bleu']:.4f}")
    print(f"ROUGE-1: {avg_metrics['rouge1']:.4f}")
    print(f"ROUGE-2: {avg_metrics['rouge2']:.4f}")
    print(f"ROUGE-L: {avg_metrics['rougeL']:.4f}")
    print(f"METEOR: {avg_metrics['meteor']:.4f}")
    
    # Visualize results
    fig, axes = plt.subplots(min(4, len(images)), 1, figsize=(10, 4*min(4, len(images))))
    if len(images) == 1:
        axes = [axes]
    
    # Denormalize images for visualization
    denorm = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    
    for i in range(min(4, len(images))):
        try:
            # Convert tensor to numpy and transpose to (H,W,C)
            img = denorm(images[i]).cpu().numpy().transpose(1, 2, 0)
            img = np.clip(img, 0, 1)
            
            # Plot image
            axes[i].imshow(img)
            axes[i].set_title(f"Ground Truth: {ground_truth_captions[i]}\nPrediction: {predicted_captions[i]}")
            axes[i].set_xticks([])
            axes[i].set_yticks([])
        except Exception as e:
            print(f"Error visualizing sample {i}: {str(e)}")
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(args['checkpoint_dir'], exist_ok=True)
    plt.savefig(os.path.join(args['checkpoint_dir'], 'inference_visualization.png'))
    plt.close()
    
    return avg_metrics

# ====================
# EVALUATION METRICS - UNCHANGED
# ====================
def calculate_bleu(reference, hypothesis):
    """Calculate BLEU score between reference and hypothesis."""
    smoothie = SmoothingFunction().method1
    reference_tokens = nltk.word_tokenize(reference.lower())
    hypothesis_tokens = nltk.word_tokenize(hypothesis.lower())
    return sentence_bleu([reference_tokens], hypothesis_tokens, smoothing_function=smoothie)

def calculate_rouge(reference, hypothesis):
    """Calculate ROUGE scores between reference and hypothesis."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure
    }

def calculate_meteor(reference, hypothesis):
    """Calculate METEOR score between reference and hypothesis."""
    reference_tokens = nltk.word_tokenize(reference.lower())
    hypothesis_tokens = nltk.word_tokenize(hypothesis.lower())
    return meteor_score([reference_tokens], hypothesis_tokens)

# ====================
# MAIN FUNCTION - RESTRUCTURED
# ====================
def main():
    print(f"Starting training with device: {device}")
    
    print("Creating data loaders...")
    # Create minimal filters
    caption_filters = [{'field': 'label', 'string_list': ['radiology']}]
    
    # Create data loaders
    train_loader = get_multicare_dataloader(
        dataset_name=f"{args['dataset']}_train",
        batch_size=args['batch_size'],
        create_new=args['create_new_dataset'],
        filters=caption_filters,
        shuffle=True
    )
    
    val_loader = get_multicare_dataloader(
        dataset_name=f"{args['dataset']}_val",
        batch_size=args['batch_size'],
        create_new=args['create_new_dataset'],
        filters=caption_filters,
        shuffle=False
    )
    
    print("Initializing models...")
    # Create encoder and decoder
    encoder = ResNetEncoder(embed_size=args['encoder_embed_size']).to(device)
    decoder = ProjectionDecoder(
        model_name=args['transformer_model'],
        image_embed_size=args['encoder_embed_size']
    ).to(device)
    
    # Set up loss function
    loss_fn = ContrastiveLoss(margin=args['margin'], temperature=args['temperature']).to(device)
    
    # Create optimizer
    print("Setting up optimizer...")
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.AdamW(params, lr=args['learning_rate'], weight_decay=args['weight_decay'])
    
    # Create scheduler with warmup
    total_steps = len(train_loader) * args['epochs']
    warmup_steps = len(train_loader) * args['warmup_epochs']
    
    def lr_lambda(current_step):
        # Linear warmup followed by cosine decay
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args['resume'] and os.path.isfile(args['resume']):
        print(f"Loading checkpoint from {args['resume']}")
        checkpoint = torch.load(args['resume'], map_location=device)
        encoder.load_state_dict(checkpoint['encoder'])
        decoder.load_state_dict(checkpoint['decoder'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    print("Starting training...")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, args['epochs']):
        # Train for one epoch
        train_loss = train_epoch(encoder, decoder, train_loader, optimizer, epoch, loss_fn, scheduler)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(encoder, decoder, val_loader, loss_fn)
        val_losses.append(val_loss)
        
        # Print progress
        print(f"Epoch {epoch+1}/{args['epochs']} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        best_val_loss = min(val_loss, best_val_loss)
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(args['checkpoint_dir'], exist_ok=True)
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'args': args
        }
        
        torch.save(checkpoint, os.path.join(args['checkpoint_dir'], 'latest_checkpoint.pt'))
        
        if is_best:
            torch.save(checkpoint, os.path.join(args['checkpoint_dir'], 'best_checkpoint.pt'))
            print(f"Saved new best model with validation loss: {val_loss:.4f}")
            
        if (epoch + 1) % args['checkpoint_freq'] == 0:
            torch.save(checkpoint, os.path.join(args['checkpoint_dir'], f'checkpoint_epoch_{epoch+1}.pt'))
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(args['checkpoint_dir'], 'loss_curve.png'))
    
    # Run inference
    print("\nRunning inference and evaluation after training...")
    run_inference_and_visualize(encoder, decoder, train_loader)
    
    print("Training completed!")

if __name__ == "__main__":
    main() 