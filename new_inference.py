import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import argparse
import matplotlib.pyplot as plt
from main import ResNetEncoder, ProjectionDecoder, MedicalImageCaptionDataset, args

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Image preprocessing
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# For denormalizing the image for display
denorm = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
)

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image_transform(image).unsqueeze(0)  # Shape: [1, 3, 224, 224]
    return image.to(device)

def cosine_similarity_matrix(vec, matrix):
    vec = vec / vec.norm(dim=-1, keepdim=True)
    matrix = matrix / matrix.norm(dim=-1, keepdim=True)
    return torch.matmul(vec, matrix.T)

def main_inference(image_path, top_k=5):
    # Load models
    encoder = ResNetEncoder(embed_size=args['encoder_embed_size']).to(device)
    decoder = ProjectionDecoder(
        model_name=args['transformer_model'],
        image_embed_size=args['encoder_embed_size']
    ).to(device)

    # Load checkpoint
    checkpoint_path = os.path.join(args['checkpoint_dir'], 'best_checkpoint.pt')
    if not os.path.isfile(checkpoint_path):
        checkpoint_path = os.path.join(args['checkpoint_dir'], 'latest_checkpoint.pt')
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])

    encoder.eval()
    decoder.eval()

    # Load dataset
    dataset = MedicalImageCaptionDataset(f"data/{args['dataset']}")
    captions = [sample['caption'] for sample in dataset]

    print(f"Loaded {len(captions)} captions from dataset.")

    # Compute text embeddings
    print("Computing text embeddings...")
    text_embeddings = torch.stack([
        decoder.encode_text(c, decoder.tokenizer).squeeze().detach()
        for c in captions
    ])
    text_embeddings = F.normalize(text_embeddings, dim=1).to(device)

    # Load and process input image
    image_tensor = load_image(image_path)

    with torch.no_grad():
        img_feat = encoder(image_tensor)
        img_embedding = decoder(img_feat)
        img_embedding = F.normalize(img_embedding, dim=1)

        # Compute similarity
        similarities = cosine_similarity_matrix(img_embedding, text_embeddings).squeeze()

    # Get top-k captions
    topk_sim, topk_indices = torch.topk(similarities, top_k)
    predicted_captions = [captions[idx.item()] for idx in topk_indices]

    print(f"\nTop {top_k} most similar captions for:\n{image_path}\n")
    for i in range(top_k):
        print(f"[{topk_sim[i].item():.4f}] {predicted_captions[i]}")

    # Visualize image with top-1 caption
    image = image_tensor.squeeze().cpu()
    image = denorm(image).clamp(0, 1).permute(1, 2, 0).numpy()

    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.axis("off")
    plt.title(f"Top Prediction:\n{predicted_captions[0]}", fontsize=12)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top captions to return')
    args_cli = parser.parse_args()

    main_inference(args_cli.image_path, top_k=args_cli.top_k)
