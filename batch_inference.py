import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from PIL import Image
from torchvision import transforms
import matplotlib.gridspec as gridspec
from datetime import datetime

# Import from main.py
from main import (
    ResNetEncoder, 
    ProjectionDecoder, 
    device, 
    get_multicare_dataloader,
    args as default_args
)

def parse_args():
    parser = argparse.ArgumentParser(description='Medical Image Caption Batch Inference')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_checkpoint.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for inference')
    parser.add_argument('--dataset', type=str, default='medCapAll2_train',
                        help='Dataset name to use (from main.py)')
    parser.add_argument('--output_dir', type=str, default='batch_inference_results',
                        help='Directory to save inference results')
    parser.add_argument('--sample_captions', type=str, default=None, 
                        help='Path to a text file with sample captions for retrieval')
    return parser.parse_args()

def load_model(checkpoint_path):
    """Load trained encoder and decoder models."""
    print(f"Loading model from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint['args']
    
    # Initialize models
    encoder = ResNetEncoder(embed_size=args['encoder_embed_size']).to(device)
    decoder = ProjectionDecoder(
        model_name=args['transformer_model'],
        image_embed_size=args['encoder_embed_size']
    ).to(device)
    
    # Load state dictionaries
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    
    # Set models to evaluation mode
    encoder.eval()
    decoder.eval()
    
    print(f"Models loaded successfully! Trained for {checkpoint['epoch']+1} epochs.")
    print(f"Final training loss: {checkpoint.get('train_loss', 'N/A')}")
    print(f"Final validation loss: {checkpoint.get('val_loss', 'N/A')}")
    
    return encoder, decoder, args

def get_sample_captions(file_path=None):
    """Get sample captions for retrieval-based inference."""
    if file_path and os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return [line.strip() for line in f.readlines() if line.strip()]
    
    # Default medical captions if no file provided
    return [
        "Normal chest radiograph with no significant abnormalities.",
        "Bilateral pulmonary infiltrates consistent with pneumonia.",
        "Cardiomegaly with signs of pulmonary congestion.",
        "Small right pleural effusion with no pneumothorax.",
        "Lung nodule in the right upper lobe requiring follow-up.",
        "Left lower lobe consolidation suggestive of pneumonia.",
        "Hyperinflation of lungs consistent with COPD.",
        "No acute cardiopulmonary process identified.",
        "Right-sided chest tube with resolved pneumothorax.",
        "Diffuse interstitial changes consistent with pulmonary fibrosis.",
        "Prominent hilar lymphadenopathy requiring further evaluation.",
        "Multiple rib fractures on the left side with no pneumothorax.",
        "Enlargement of the cardiac silhouette suggesting cardiomegaly.",
        "Endotracheal tube appropriately positioned.",
        "Mild degenerative changes in the thoracic spine.",
        "Evidence of pulmonary edema with perihilar distribution.",
        "Calcified granuloma in the right upper lobe, likely old infection.",
        "Bilateral hilar prominence possibly representing lymphadenopathy.",
        "Blunting of the costophrenic angle suggestive of small pleural effusion.",
        "No evidence of pneumothorax or pleural effusion.",
        "Patchy opacities in both lung fields consistent with atypical infection.",
        "Clear lung fields with no focal consolidation.",
        "Increased opacity in the left lower lobe concerning for pneumonia.",
        "Bilateral interstitial opacities consistent with pulmonary edema.",
        "Tracheal deviation to the right due to left-sided tension pneumothorax."
    ]

def run_batch_inference(encoder, decoder, batch, sample_captions):
    """Run inference on a batch of images."""
    # Get images and captions from the batch
    images = batch['image'].to(device)
    original_captions = batch['caption']
    
    # List to store results for each image
    results = []
    
    with torch.no_grad():
        # Get image features and embeddings for the whole batch
        image_features = encoder(images)
        image_embeddings = decoder(image_features)
        
        # Normalize embeddings
        image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
        
        # Get embeddings for all sample captions (only need to do this once)
        text_embeddings = []
        for caption in sample_captions:
            text_emb = decoder.encode_text(caption, decoder.tokenizer)
            text_embeddings.append(F.normalize(text_emb.squeeze(), p=2, dim=0))
        
        # Stack text embeddings
        text_embeddings = torch.stack(text_embeddings)
        
        # Compute similarity for the whole batch
        similarity = torch.matmul(image_embeddings, text_embeddings.T)
        
        # For each image in the batch
        for i in range(len(images)):
            # Get top 3 most similar captions
            img_similarity = similarity[i].cpu().numpy()
            top_indices = np.argsort(img_similarity)[::-1][:3]
            
            top_captions = [(sample_captions[idx], img_similarity[idx]) for idx in top_indices]
            
            # Store results
            results.append({
                'image': images[i],
                'original_caption': original_captions[i],
                'predictions': top_captions
            })
        
    return results

def denormalize_image(image_tensor):
    """Convert normalized tensor to numpy image for display."""
    # Denormalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image_tensor.cpu() * std + mean
    
    # Convert to numpy and transpose to (H,W,C)
    image = image.numpy().transpose(1, 2, 0)
    
    # Clip values to valid range
    image = np.clip(image, 0, 1)
    
    return image

def visualize_batch_results(results, save_path=None):
    """Visualize inference results for a batch of images."""
    batch_size = len(results)
    
    # Create figure with a grid layout
    fig = plt.figure(figsize=(16, 4 * batch_size))
    gs = gridspec.GridSpec(batch_size, 2, width_ratios=[1, 1.5])
    
    for i, result in enumerate(results):
        # Left subplot: Image
        ax_img = plt.subplot(gs[i, 0])
        image = denormalize_image(result['image'])
        ax_img.imshow(image)
        ax_img.set_title(f"Image {i+1}")
        ax_img.axis('off')
        
        # Right subplot: Captions
        ax_txt = plt.subplot(gs[i, 1])
        ax_txt.axis('off')
        
        # Display original caption
        original_caption = result['original_caption']
        ax_txt.text(0.02, 0.95, "Original Caption:", ha='left', va='top',
                   fontsize=12, fontweight='bold', color='green')
        
        # Wrap long captions
        wrapped_original = "\n".join([original_caption[j:j+60] for j in range(0, len(original_caption), 60)])
        ax_txt.text(0.02, 0.85, wrapped_original, ha='left', va='top',
                   fontsize=11, color='green')
        
        # Display predictions
        ax_txt.text(0.02, 0.65, "Model Predictions:", ha='left', va='top',
                   fontsize=12, fontweight='bold', color='blue')
        
        for j, (caption, score) in enumerate(result['predictions']):
            # Wrap long captions
            wrapped_caption = "\n".join([caption[k:k+60] for k in range(0, len(caption), 60)])
            y_pos = 0.58 - j*0.15
            ax_txt.text(0.02, y_pos, f"{j+1}. {wrapped_caption}", ha='left', va='top',
                       fontsize=10)
            ax_txt.text(0.02, y_pos-0.05, f"Confidence: {score:.4f}", ha='left', va='top',
                       fontsize=9, color='blue')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    
    plt.show()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    encoder, decoder, model_args = load_model(args.checkpoint)
    
    # Get sample captions
    sample_captions = get_sample_captions(args.sample_captions)
    print(f"Loaded {len(sample_captions)} sample captions for retrieval")
    
    # Create data loader
    minimal_filters = [{'field': 'label', 'string_list': ['radiology']}]
    
    dataloader = get_multicare_dataloader(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        create_new=False,  # Don't recreate the dataset
        filters=minimal_filters,
        shuffle=True  # Shuffle to get random batch
    )
    
    print(f"Created dataloader for dataset: {args.dataset}")
    
    # Get a random batch
    dataloader_iter = iter(dataloader)
    batch = next(dataloader_iter)
    
    print(f"Got random batch with {len(batch['image'])} images")
    
    # Run inference on the batch
    results = run_batch_inference(encoder, decoder, batch, sample_captions)
    
    # Visualize results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    visualize_batch_results(
        results,
        save_path=os.path.join(args.output_dir, f"batch_inference_{timestamp}.png")
    )
    
    print(f"Inference completed. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 