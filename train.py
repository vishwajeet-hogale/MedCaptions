"""
Simple training script for medical image captioning with LSTM decoder.

This is a basic training script for quick experimentation and prototyping.
For more advanced features, use train_medical_caption.py instead.
"""
import torch
from transformers import BertTokenizer, BertModel
from caption_lstm import CaptionLSTM
from deit_encoder import DeiTMedicalEncoder
from dataloader import get_multicare_dataloader
from tqdm import tqdm
import os

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
os.makedirs("checkpoints", exist_ok=True) 

# Load tokenizer & BERT
tokenizer = BertTokenizer.from_pretrained("MediCareBertTokenizer")
bert = BertModel.from_pretrained("MediCareBertModel").to(device).eval()

# Models
decoder = CaptionLSTM().to(device)
encoder = DeiTMedicalEncoder().to(device)

# Freeze BERT
for param in bert.parameters():
    param.requires_grad = False

# Optimizer
optimizer = torch.optim.Adam(
    list(decoder.parameters()) + list(encoder.parameters()), lr=1e-4
)

# Data (no pin_memory for MPS)
loader = get_multicare_dataloader(
    dataset_name='brain_tumor_mri_1742661954',
    batch_size=8,
    create_new=False,
    filters=[
        {'field': 'label', 'string_list': ['mri', 'head']},
        {'field': 'caption', 'string_list': ['showing', 'demonstrates', 'reveals'], 'operator': 'any'}
    ],
    num_workers=0  # MPS sometimes doesn’t like multiprocessing
)

def get_bert_embeddings(captions):
    """
    Generate BERT embeddings for a batch of captions.
    
    Args:
        captions (list): List of caption strings
        
    Returns:
        torch.Tensor: BERT embeddings [batch, seq_len, 768]
    """
    inputs = tokenizer(captions, return_tensors='pt', truncation=True,
                       padding='max_length', max_length=256).to(device)
    with torch.no_grad():
        out = bert(**inputs)
    return out.last_hidden_state


def train_caption_model(epochs=5):
    """
    Train the caption generation model.
    
    Args:
        epochs (int): Number of training epochs
    """
    decoder.train()
    encoder.train()

    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}"):
            images = batch['image'].to(device)
            captions = batch['caption']

            # Forward pass
            image_feats = encoder(images)
            bert_embed = get_bert_embeddings(captions)
            pred = decoder(bert_embed, image_feats)
            target = bert_embed[:, 0, :]  # CLS token

            # Compute loss
            loss = torch.nn.functional.mse_loss(pred, target)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg = total_loss / len(loader)
        print(f"Epoch {epoch+1} - Avg Loss: {avg:.4f}")

        # Save checkpoint
        torch.save({
            'decoder': decoder.state_dict(),
            'encoder': encoder.state_dict(),
            'optimizer': optimizer.state_dict()
        }, f"checkpoints/checkpoint_epoch_{epoch+1}.pt")


if __name__ == "__main__":
    train_caption_model()
