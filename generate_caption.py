import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from caption_lstm import CaptionLSTM
from deit_encoder import DeiTMedicalEncoder
from dataloader import get_multicare_dataloader  # Adjust if needed
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

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
# LOAD CAPTION LSTM DECODER
# ====================
# Initialize models
decoder = CaptionLSTM(hidden_size=1024, num_layers=2)
encoder = DeiTMedicalEncoder(embed_size=768)

# Load checkpoint
checkpoint = torch.load("checkpoints/checkpoint_epoch_25.pt", map_location=device)

# Load weights
decoder.load_state_dict(checkpoint["decoder"])
encoder.load_state_dict(checkpoint["encoder"])

# Move to device and eval mode
decoder.to(device).eval()
encoder.to(device).eval()



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
def test_captioning_batch_from_itself(loader, top_k=3):
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
    for i in range(min(4, len(images))):  # Limit to 4 samples for display
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
# MAIN EXECUTION
# ====================
if __name__ == "_main_":
    # Define dataset filters
    caption_filters = [
        {'field': 'label', 'string_list': ['mri', 'head']},
        {'field': 'caption', 'string_list': ['showing', 'demonstrates', 'reveals'], 'operator': 'any'}
    ]

    # Load dataloader
    loader = get_multicare_dataloader(
        dataset_name='med_test',
        batch_size=4,
        create_new=False,
        filters=caption_filters
    )

    # Run test
    test_captioning_batch_from_itself(loader, top_k=3)