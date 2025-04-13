import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, AutoTokenizer
from transformer_decoder import TransformerMedicalDecoder
from deit_encoder import DeiTMedicalEncoder
from dataloader import get_multicare_dataloader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import argparse

# ====================
# DEVICE SETUP
# ====================
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ====================
# LOAD BERT MODEL
# ====================
tokenizer = BertTokenizer.from_pretrained("./MediCareBertTokenizer")
bert_model = BertModel.from_pretrained("./MediCareBertModel").to(device).eval()

def get_caption_embedding(caption):
    """Generate BERT CLS embedding for a given caption."""
    inputs = tokenizer(caption, return_tensors='pt', truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :]  # CLS token embedding

# ====================
# LOAD TRANSFORMER DECODER & IMAGE ENCODER
# ====================
def load_models(checkpoint_path):
    # Initialize encoder
    encoder = DeiTMedicalEncoder(embed_size=768)
    
    # Initialize the new transformer decoder
    decoder = TransformerMedicalDecoder(
        image_embed_size=384,  # Match DeiT encoder output
        freeze_base=False  # Allow full fine-tuning for inference
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load weights
    try:
        # Try loading with new decoder keys
        decoder.load_state_dict(checkpoint["decoder"])
    except:
        print("Checkpoint was saved with LSTM decoder. Using transformer in evaluation mode only.")
        # If using a checkpoint from the LSTM model, we'll just use the transformer in eval mode
        
    encoder.load_state_dict(checkpoint["encoder"])
    
    # Move to device and eval mode
    decoder.to(device).eval()
    encoder.to(device).eval()
    
    return encoder, decoder

# ====================
# INVERSE TRANSFORM FOR VISUALIZATION
# ====================
inv_normalize = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
)

# ====================
# TEST FUNCTION USING BATCH CAPTIONS AS REFERENCE
# ====================
def test_captioning_batch_from_itself(encoder, decoder, loader, top_k=3, num_samples=4):
    batch = next(iter(loader))
    images = batch['image'].to(device)
    captions = batch['caption']

    print(f"\nEvaluating {len(images)} images using in-batch captions as reference...")

    # Step 1: Encode all ground truth captions
    with torch.no_grad():
        caption_embeddings = torch.stack([
            get_caption_embedding(c).squeeze() for c in captions
        ]).cpu().numpy()

    # Step 2: Predict caption embeddings from images
    predicted_embeddings = []
    for i in range(len(images)):
        with torch.no_grad():
            image_feat = encoder(images[i].unsqueeze(0))
            dummy_input = torch.zeros((1, 1, 768)).to(device)
            pred_embed = decoder(dummy_input, image_feat).squeeze().cpu().numpy()
            predicted_embeddings.append(pred_embed)

    # Step 3: Compare each prediction to all in-batch caption embeddings
    for i in range(min(num_samples, len(images))):  # Limit to specified samples for display
        pred_embed = predicted_embeddings[i]

        sims = F.cosine_similarity(
            torch.tensor(pred_embed).unsqueeze(0),
            torch.tensor(caption_embeddings),
            dim=1
        ).numpy()

        top_indices = sims.argsort()[-top_k:][::-1]

        print(f"\nImage {i+1}")
        print(f"Ground Truth: {captions[i]}")
        print(f"Top-{top_k} Similar Captions (from batch):")
        for j, idx in enumerate(top_indices):
            print(f"   {j+1}. {captions[idx]} (sim: {sims[idx]:.4f})")

        # Optional: Visualize the image
        img_np = inv_normalize(images[i].cpu()).permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)
        plt.imshow(img_np)
        plt.title(f"Sample {i+1}")
        plt.axis("off")
        plt.show()

# ====================
# COMPARE PERFORMANCE WITH LSTM MODEL
# ====================
def compare_with_lstm(image, lstm_decoder, transformer_decoder, encoder, captions):
    """Compare the performance of the LSTM and transformer decoders on a single image"""
    from caption_lstm import CaptionLSTM
    
    # Extract features
    with torch.no_grad():
        image_feat = encoder(image.unsqueeze(0).to(device))
        dummy_input = torch.zeros((1, 1, 768)).to(device)
        
        # Get predictions from both models
        lstm_embed = lstm_decoder(dummy_input, image_feat).squeeze().cpu().numpy()
        transformer_embed = transformer_decoder(dummy_input, image_feat).squeeze().cpu().numpy()
        
        # Encode ground truth captions
        caption_embeddings = torch.stack([
            get_caption_embedding(c).squeeze() for c in captions
        ]).cpu().numpy()
        
        # Calculate similarities
        lstm_sims = F.cosine_similarity(
            torch.tensor(lstm_embed).unsqueeze(0),
            torch.tensor(caption_embeddings),
            dim=1
        ).numpy()
        
        transformer_sims = F.cosine_similarity(
            torch.tensor(transformer_embed).unsqueeze(0),
            torch.tensor(caption_embeddings),
            dim=1
        ).numpy()
        
        # Get top results
        lstm_top_indices = lstm_sims.argsort()[-3:][::-1]
        transformer_top_indices = transformer_sims.argsort()[-3:][::-1]
        
        # Display results
        print("\nLSTM Results:")
        for j, idx in enumerate(lstm_top_indices):
            print(f"   {j+1}. {captions[idx]} (sim: {lstm_sims[idx]:.4f})")
            
        print("\nTransformer Results:")
        for j, idx in enumerate(transformer_top_indices):
            print(f"   {j+1}. {captions[idx]} (sim: {transformer_sims[idx]:.4f})")
        
        # Show image
        img_np = inv_normalize(image.cpu()).permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)
        plt.imshow(img_np)
        plt.title("Comparison Sample")
        plt.axis("off")
        plt.show()

# ====================
# MAIN EXECUTION
# ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate medical image captions using transformer')
    parser.add_argument('--checkpoint', type=str, default="checkpoints/checkpoint_epoch_25.pt", 
                        help='Path to model checkpoint')
    parser.add_argument('--compare', action='store_true', 
                        help='Compare with LSTM model performance')
    parser.add_argument('--samples', type=int, default=4, 
                        help='Number of samples to evaluate')
    parser.add_argument('--topk', type=int, default=3, 
                        help='Number of top predictions to show')
    
    args = parser.parse_args()
    
    # Define dataset filters
    caption_filters = [
        {'field': 'label', 'string_list': ['mri', 'head']},
        {'field': 'caption', 'string_list': ['showing', 'demonstrates', 'reveals'], 'operator': 'any'}
    ]

    # Load dataloader
    loader = get_multicare_dataloader(
        dataset_name='med_test',
        batch_size=args.samples,
        create_new=False,
        filters=caption_filters
    )
    
    # Load models
    encoder, transformer_decoder = load_models(args.checkpoint)
    
    # Run test with transformer decoder
    test_captioning_batch_from_itself(encoder, transformer_decoder, loader, 
                                     top_k=args.topk, num_samples=args.samples)
    
    # Optionally compare with LSTM model
    if args.compare:
        try:
            from caption_lstm import CaptionLSTM
            
            # Load LSTM model
            lstm_decoder = CaptionLSTM(hidden_size=1024, num_layers=2)
            checkpoint = torch.load(args.checkpoint, map_location=device)
            lstm_decoder.load_state_dict(checkpoint["decoder"])
            lstm_decoder.to(device).eval()
            
            # Get a batch for comparison
            batch = next(iter(loader))
            
            # Compare the first image
            compare_with_lstm(batch['image'][0], lstm_decoder, transformer_decoder, 
                              encoder, batch['caption'])
            
        except Exception as e:
            print(f"Could not compare with LSTM model: {e}") 