import torch
import nltk
from transformers import BertTokenizer, BertModel
from caption_lstm import CaptionLSTM
from deit_encoder import DeiTMedicalEncoder
from dataloader import get_multicare_dataloader
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
from rouge_score import rouge_scorer

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

# ===========================
# DEVICE SETUP
# ===========================
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ===========================
# LOAD TOKENIZER & MODELS
# ===========================
tokenizer = BertTokenizer.from_pretrained("MediCareBertTokenizer")
bert_model = BertModel.from_pretrained("MediCareBertModel").to(device).eval()

encoder = DeiTMedicalEncoder(embed_size=768)
decoder = CaptionLSTM(hidden_size=1024, num_layers=2)

checkpoint = torch.load("checkpoints/checkpoint_epoch_4.pt", map_location=device)
encoder.load_state_dict(checkpoint["encoder"])
decoder.load_state_dict(checkpoint["decoder"])

encoder.to(device).eval()
decoder.to(device).eval()

# ===========================
# PLACEHOLDER CAPTION GENERATION
# ===========================
def generate_caption(image_tensor, max_len=30):
    """Stub: Replace this with actual decoding logic."""
    with torch.no_grad():
        features = encoder(image_tensor.unsqueeze(0).to(device))
        input_token = torch.zeros((1, 1, 768)).to(device)
        _ = decoder(input_token, features)
        return "Generated caption text here."  # <- Replace with decoded output

# ===========================
# EVALUATION METRICS
# ===========================
def compute_bleu_all(reference, hypothesis):
    ref_tokens = [nltk.word_tokenize(reference.lower())]
    hyp_tokens = nltk.word_tokenize(hypothesis.lower())
    smoothie = SmoothingFunction().method4

    bleu1 = sentence_bleu(ref_tokens, hyp_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie)
    bleu2 = sentence_bleu(ref_tokens, hyp_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
    bleu3 = sentence_bleu(ref_tokens, hyp_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie)
    bleu4 = sentence_bleu(ref_tokens, hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)

    return bleu1, bleu2, bleu3, bleu4

def compute_meteor(reference, hypothesis):
    return single_meteor_score(reference, hypothesis)

def compute_rouge_all(reference, hypothesis):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure
    }

# ===========================
# MAIN EVALUATION FUNCTION
# ===========================
def evaluate(loader):
    bleu_1_all, bleu_2_all, bleu_3_all, bleu_4_all = [], [], [], []
    rouge_1_all, rouge_2_all, rouge_l_all = [], [], []
    meteor_all = []

    batch = next(iter(loader))
    images = batch["image"].to(device)
    captions = batch["caption"]

    for i in range(len(images)):
        reference = captions[i]
        hypothesis = generate_caption(images[i])

        b1, b2, b3, b4 = compute_bleu_all(reference, hypothesis)
        bleu_1_all.extend([b1])
        bleu_2_all.extend([b2])
        bleu_3_all.extend([b3])
        bleu_4_all.extend([b4])

        rouge = compute_rouge_all(reference, hypothesis)
        rouge_1_all.append(rouge['rouge1'])
        rouge_2_all.append(rouge['rouge2'])
        rouge_l_all.append(rouge['rougeL'])

        meteor = compute_meteor(reference, hypothesis)
        meteor_all.append(meteor)

        print(f"\nSample {i+1}")
        print(f"Reference:  {reference}")
        print(f"Hypothesis: {hypothesis}")
        print(f"BLEU-1: {b1:.4f} | BLEU-2: {b2:.4f} | BLEU-3: {b3:.4f} | BLEU-4: {b4:.4f}")
        print(f"ROUGE-1: {rouge['rouge1']:.4f} | ROUGE-2: {rouge['rouge2']:.4f} | ROUGE-L: {rouge['rougeL']:.4f}")
        print(f"METEOR:  {meteor:.4f}")

    print("\n=== AVERAGE METRICS ===")
    print(f"BLEU-1:  {sum(bleu_1_all)/len(bleu_1_all):.4f}")
    print(f"BLEU-2:  {sum(bleu_2_all)/len(bleu_2_all):.4f}")
    print(f"BLEU-3:  {sum(bleu_3_all)/len(bleu_3_all):.4f}")
    print(f"BLEU-4:  {sum(bleu_4_all)/len(bleu_4_all):.4f}")
    print(f"ROUGE-1: {sum(rouge_1_all)/len(rouge_1_all):.4f}")
    print(f"ROUGE-2: {sum(rouge_2_all)/len(rouge_2_all):.4f}")
    print(f"ROUGE-L: {sum(rouge_l_all)/len(rouge_l_all):.4f}")
    print(f"METEOR:  {sum(meteor_all)/len(meteor_all):.4f}")


if __name__ == "__main__":
    loader = get_multicare_dataloader(
        dataset_name='med_test',
        batch_size=4,
        create_new=False
    )
    evaluate(loader)
