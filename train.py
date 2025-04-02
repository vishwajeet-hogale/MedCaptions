import torch
from transformers import BertTokenizer, BertModel
from caption_lstm import CaptionLSTM
from deit_encoder import DeiTMedicalEncoder
from dataloader import get_multicare_dataloader
from tqdm import tqdm

# DEVICE SETUP
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Load tokenizer & BERT
tokenizer = BertTokenizer.from_pretrained("MediCareBert")
bert = BertModel.from_pretrained("MediCareBert").to(device).eval()

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
    inputs = tokenizer(captions, return_tensors='pt', truncation=True,
                       padding='max_length', max_length=256).to(device)
    with torch.no_grad():
        out = bert(**inputs)
    return out.last_hidden_state  # [batch, seq_len, 768]

def train_caption_model(epochs=1):
    decoder.train()
    encoder.train()

    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}"):
            images = batch['image'].to(device)
            captions = batch['caption']

            image_feats = encoder(images)                     # [B, 768]
            bert_embed = get_bert_embeddings(captions)        # [B, seq_len, 768]
            pred = decoder(bert_embed, image_feats)           # [B, 768]
            target = bert_embed[:, 0, :]                      # CLS token

            loss = torch.nn.functional.mse_loss(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg = total_loss / len(loader)
        print(f"Epoch {epoch+1} - Avg Loss: {avg:.4f}")

        # Save model
        torch.save({
            'decoder': decoder.state_dict(),
            'encoder': encoder.state_dict(),
            'optimizer': optimizer.state_dict()
        }, f"checkpoints/checkpoint_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    train_caption_model()
