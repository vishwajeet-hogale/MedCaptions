import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from caption_decoder import CaptionLSTM
from encoder_deit import DeiTMedicalEncoder
from torchvision import transforms
from PIL import Image
import os

# DEVICE SETUP
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Load BERT
tokenizer = BertTokenizer.from_pretrained("MediCareBert")
bert = BertModel.from_pretrained("MediCareBert").to(device).eval()

# Load Captioning Models
decoder = CaptionLSTM()
decoder.load_state_dict(torch.load("checkpoints/checkpoint_epoch_1.pt", map_location=device)["decoder"])
decoder.to(device).eval()

encoder = DeiTMedicalEncoder()
encoder.load_state_dict(torch.load("checkpoints/checkpoint_epoch_1.pt", map_location=device)["encoder"])
encoder.to(device).eval()

# Precompute vocab embeddings
def get_bert_vocab_embeddings():
    with torch.no_grad():
        vocab_size = tokenizer.vocab_size
        tokens = [tokenizer.convert_ids_to_tokens(i) for i in range(vocab_size)]
        encoded = tokenizer(tokens, return_tensors='pt', padding=True, truncation=True).to(device)
        outputs = bert(**encoded)
        embeddings = outputs.last_hidden_state[:, 0, :]
    return tokens, embeddings  # [vocab_size, 768]

tokens_list, vocab_embeddings = get_bert_vocab_embeddings()

def generate_caption(image_path, max_len=20, temperature=0.0):
    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    image_feat = encoder(image)  # [1, 768]

    generated_tokens = ["[CLS]"]

    for _ in range(max_len):
        current_input = tokenizer(
            " ".join(generated_tokens),
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            max_length=256
        ).to(device)

        with torch.no_grad():
            out = bert(**current_input)
            bert_embed = out.last_hidden_state  # [1, seq_len, 768]
            pred = decoder(bert_embed, image_feat)  # [1, 768]

            # Cosine similarity
            norm_pred = F.normalize(pred, dim=1)
            norm_vocab = F.normalize(vocab_embeddings, dim=1)
            sims = torch.matmul(norm_pred, norm_vocab.T).squeeze()

            if temperature == 0:
                next_idx = sims.argmax().item()
            else:
                probs = F.softmax(sims / temperature, dim=0)
                next_idx = torch.multinomial(probs, 1).item()

            next_token = tokens_list[next_idx]

            if next_token in ["[SEP]", "[PAD]", "[CLS]"]:
                break

            generated_tokens.append(next_token)

    return tokenizer.convert_tokens_to_string(generated_tokens[1:])

# Test
if __name__ == "__main__":
    img_path = "medical_datasets/brain_tumor_mri/images/sample1.jpg"
    if os.path.exists(img_path):
        print("Generated Caption:", generate_caption(img_path))
    else:
        print("Image not found, please update path.")
