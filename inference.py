import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import json
import re
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import glob
import random

# Import model architecture from main.py
from main import ResNetEncoder, ProjectionDecoder, device

def parse_args():
    parser = argparse.ArgumentParser(description='Medical Image Caption Inference')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/checkpoint_epoch_30.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--image_dir', type=str, default=None,
                        help='Directory containing images for inference')
    parser.add_argument('--image_path', type=str, default=None,
                        help='Path to a single image for inference')
    parser.add_argument('--output_dir', type=str, default='inference_results',
                        help='Directory to save inference results')
    parser.add_argument('--sample_captions', type=str, default=None, 
                        help='Path to a text file with sample captions for retrieval')
    parser.add_argument('--captions_json', type=str, default=None,
                        help='Path to JSON file mapping image filenames to their original captions')
    return parser.parse_args()

def load_caption_mapping(json_path):
    """Load mapping of image filenames to their original captions."""
    if json_path and os.path.exists(json_path):
        with open(json_path, 'r') as f:
            return json.load(f)
    return {}

def get_original_caption(image_path, caption_mapping):
    """Get the original caption for an image."""
    # Try to find the caption in the mapping
    filename = os.path.basename(image_path)
    if filename in caption_mapping:
        return caption_mapping[filename]
    
    # If no mapping is provided, try to extract info from the filename
    # This is a fallback assuming filenames might contain some information
    parts = re.split(r'[_\-.]', filename)
    if len(parts) > 2:
        return f"Medical image: {' '.join(parts[:-1])}"
    
    # Default if no caption can be found
    return "Original caption not available"

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
        "Tracheal deviation to the right due to left-sided tension pneumothorax.",
        # Additional diverse captions
        "Abdominal CT showing hepatic mass with arterial enhancement.",
        "Brain MRI demonstrating multiple hyperintense lesions in white matter.",
        "Ultrasound of thyroid reveals a hypoechoic nodule with irregular margins.",
        "Mammogram showing clustered microcalcifications in the upper outer quadrant.",
        "CT angiogram of the chest demonstrating pulmonary embolism.",
        "MRI of the lumbar spine revealing disc herniation at L4-L5 level.",
        "Bone scan showing increased uptake in the right femur suggestive of fracture.",
        "Coronary angiogram demonstrating significant stenosis in the left anterior descending artery.",
        "Renal ultrasound showing hydronephrosis with proximal ureteric stone.",
        "PET-CT showing hypermetabolic lesion in the liver concerning for metastasis.",
        "CT of abdomen revealing acute appendicitis with periappendiceal inflammation.",
        "Carotid Doppler ultrasound showing significant stenosis at the bifurcation.",
        "Echocardiogram demonstrating reduced left ventricular ejection fraction.",
        "MRI of knee showing complete tear of the anterior cruciate ligament.",
        "Gallbladder ultrasound revealing multiple gallstones without wall thickening.",
        "Chest X-ray showing right middle lobe atelectasis.",
        "CT head revealing acute subarachnoid hemorrhage in the basal cisterns.",
        "Pelvic ultrasound showing complex ovarian cyst with internal echoes.",
        "Chest CT showing ground glass opacities consistent with COVID-19 pneumonia.",
        "Barium swallow demonstrating hiatal hernia with gastroesophageal reflux."
    ]

def process_image(image_path):
    """Load and preprocess an image for model inference."""
    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and transform image
    try:
        image = Image.open(image_path).convert('RGB')
        transformed_image = transform(image)
        return transformed_image, image
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None, None

def run_inference(encoder, decoder, image_tensor, sample_captions):
    """Run inference on a single image."""
    # Move image to device and add batch dimension
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Get image features and embeddings
        image_features = encoder(image_tensor)
        image_embeddings = decoder(image_features)
        
        # Normalize embeddings
        image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
        
        # Get embeddings for all sample captions
        text_embeddings = []
        for caption in sample_captions:
            text_emb = decoder.encode_text(caption, decoder.tokenizer)
            text_embeddings.append(F.normalize(text_emb.squeeze(), p=2, dim=0))
        
        # Stack text embeddings and compute similarity
        text_embeddings = torch.stack(text_embeddings)
        similarity = torch.matmul(image_embeddings, text_embeddings.T)
        
        # Get top 5 most similar captions instead of just top 3
        similarity = similarity.cpu().numpy()[0]
        top_indices = np.argsort(similarity)[::-1][:5]
        
        # Add some randomness - choose 3 captions from the top 5
        selected_indices = random.sample(top_indices.tolist(), min(3, len(top_indices)))
        
        # Create a list of tuples with (caption, similarity score)
        # Sort by similarity score to maintain the ranked order
        top_captions = [(sample_captions[idx], similarity[idx]) for idx in selected_indices]
        top_captions.sort(key=lambda x: x[1], reverse=True)
        
    return top_captions

def visualize_results(image, top_captions, original_caption, save_path=None):
    """Visualize the image with original and predicted captions."""
    plt.figure(figsize=(14, 10))  # Increased figure size
    
    # Display image - make it larger
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.axis('off')
    plt.title('Input Image', fontsize=16)
    
    # Display captions
    plt.subplot(1, 2, 2)
    plt.axis('off')
    
    # Display original caption
    plt.text(0.5, 0.95, "Original Caption:", ha='center', va='center', 
             fontsize=16, fontweight='bold', color='green')
    # Wrap long captions
    wrapped_original = "\n".join([original_caption[i:i+60] for i in range(0, len(original_caption), 60)])
    plt.text(0.5, 0.87, wrapped_original, ha='center', va='center', 
             fontsize=14, fontweight='normal', color='green')
    
    # Display predicted captions
    plt.text(0.5, 0.75, "Model Predictions:", ha='center', va='center', 
             fontsize=16, fontweight='bold', color='blue')
    
    # Increase spacing between captions
    for i, (caption, score) in enumerate(top_captions):
        # Wrap long captions
        wrapped_caption = "\n".join([caption[j:j+60] for j in range(0, len(caption), 60)])
        # Use more spacing between captions (0.2 instead of 0.15)
        y_pos = 0.65 - i*0.2  
        plt.text(0.5, y_pos, f"{i+1}. {wrapped_caption}", ha='center', va='center', 
                fontsize=13, fontweight='medium')
        plt.text(0.5, y_pos-0.06, f"Confidence: {score:.4f}", ha='center', va='center', 
                 fontsize=11, color='blue')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)  # Higher resolution
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
    
    # Load caption mapping if provided
    caption_mapping = load_caption_mapping(args.captions_json)
    if args.captions_json:
        print(f"Loaded caption mapping with {len(caption_mapping)} entries")
    
    # Process single image
    if args.image_path:
        # Process the image
        image_tensor, original_image = process_image(args.image_path)
        if image_tensor is not None:
            # Get original caption
            original_caption = get_original_caption(args.image_path, caption_mapping)
            
            # Run inference
            top_captions = run_inference(encoder, decoder, image_tensor, sample_captions)
            
            # Visualize results
            visualize_results(
                original_image,
                top_captions,
                original_caption,
                save_path=os.path.join(args.output_dir, f"result_{os.path.basename(args.image_path)}")
            )
    
    # Process directory of images
    elif args.image_dir:
        # Get all image files in directory
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
            image_files.extend(glob.glob(os.path.join(args.image_dir, ext)))
        
        print(f"Found {len(image_files)} images in {args.image_dir}")
        
        # Process each image
        for image_path in tqdm(image_files, desc="Processing images"):
            # Process the image
            image_tensor, original_image = process_image(image_path)
            if image_tensor is not None:
                # Get original caption
                original_caption = get_original_caption(image_path, caption_mapping)
                
                # Run inference
                top_captions = run_inference(encoder, decoder, image_tensor, sample_captions)
                
                # Visualize results
                visualize_results(
                    original_image,
                    top_captions,
                    original_caption,
                    save_path=os.path.join(args.output_dir, f"result_{os.path.basename(image_path)}")
                )
    
    else:
        print("Please provide either --image_path or --image_dir")
        print("To include original captions, provide a JSON file with --captions_json")
        print("The JSON should map image filenames to their captions")

if __name__ == "__main__":
    main() 