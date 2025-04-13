import torch
import torch.nn.functional as F
import nltk
from transformers import BertTokenizer, BertModel
from caption_lstm import CaptionLSTM
from deit_encoder import DeiTMedicalEncoder
from dataloader import get_multicare_dataloader
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
#from rouge_score import rouge_scorer
from tqdm import tqdm
nltk.download('punkt_tab')
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
tokenizer = BertTokenizer.from_pretrained("./MediCareBertTokenizer")
bert_model = BertModel.from_pretrained("./MediCareBertModel").to(device).eval()

encoder = DeiTMedicalEncoder(embed_size=768)
decoder = CaptionLSTM(hidden_size=1024, num_layers=2)

checkpoint = torch.load("checkpoints/checkpoint_epoch_25.pt", map_location=device)
encoder.load_state_dict(checkpoint["encoder"])
decoder.load_state_dict(checkpoint["decoder"])

encoder.to(device).eval()
decoder.to(device).eval()

# ===========================
# CAPTION GENERATION
# ===========================
def get_caption_embedding(caption):
    """Generate BERT CLS embedding for a given caption."""
    inputs = tokenizer(caption, return_tensors='pt', truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze(0)  # Return 1D tensor

def generate_caption(image_tensor, max_len=30):
    """Generate a caption for the given image tensor."""
    with torch.no_grad():
        features = encoder(image_tensor.unsqueeze(0).to(device))
        dummy_input = torch.zeros((1, 1, 768)).to(device)
        pred_embedding = decoder(dummy_input, features).squeeze()  # Make sure it's a 1D tensor
        
        # Create reference loader with small batch size
        reference_loader = get_multicare_dataloader(
            dataset_name='med_test',
            batch_size=32,
            create_new=False
        )
        
        # Get a batch of reference captions
        reference_batch = next(iter(reference_loader))
        reference_captions = reference_batch['caption']
        
        # Find the closest caption using embedding similarity
        best_similarity = -float('inf')
        best_caption = "No caption found"
        
        for caption in reference_captions:
            caption_embedding = get_caption_embedding(caption)
            
            # Make sure both tensors are 1D for proper comparison
            if len(pred_embedding.shape) > 1:
                pred_emb_1d = pred_embedding.squeeze()
            else:
                pred_emb_1d = pred_embedding
                
            if len(caption_embedding.shape) > 1:
                caption_emb_1d = caption_embedding.squeeze() 
            else:
                caption_emb_1d = caption_embedding
            
            # Calculate similarity using 1D tensors
            similarity = F.cosine_similarity(
                pred_emb_1d.unsqueeze(0), 
                caption_emb_1d.unsqueeze(0)
            ).item()
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_caption = caption
        
        return best_caption

# ===========================
# BLEU COMPUTATION FUNCTIONS
# ===========================
def compute_bleu(reference, hypothesis):
    """Compute BLEU score between reference and hypothesis."""
    reference_tokens = nltk.word_tokenize(reference.lower())
    hypothesis_tokens = nltk.word_tokenize(hypothesis.lower())
    references = [reference_tokens]  # BLEU expects a list of references
    
    smoothie = SmoothingFunction().method1
    return sentence_bleu(references, hypothesis_tokens, smoothing_function=smoothie)

def compute_bleu_all(reference, hypothesis):
    """Compute BLEU-1,2,3,4 scores."""
    reference_tokens = nltk.word_tokenize(reference.lower())
    hypothesis_tokens = nltk.word_tokenize(hypothesis.lower())
    references = [reference_tokens]
    smoothie = SmoothingFunction().method1
    
    weights1 = (1.0, 0.0, 0.0, 0.0)
    weights2 = (0.5, 0.5, 0.0, 0.0)
    weights3 = (0.33, 0.33, 0.33, 0.0)
    weights4 = (0.25, 0.25, 0.25, 0.25)
    
    b1 = sentence_bleu(references, hypothesis_tokens, weights=weights1, smoothing_function=smoothie)
    b2 = sentence_bleu(references, hypothesis_tokens, weights=weights2, smoothing_function=smoothie)
    b3 = sentence_bleu(references, hypothesis_tokens, weights=weights3, smoothing_function=smoothie)
    b4 = sentence_bleu(references, hypothesis_tokens, weights=weights4, smoothing_function=smoothie)
    
    return b1, b2, b3, b4

# ===========================
# METEOR COMPUTATION
# ===========================
def compute_meteor(reference, hypothesis):
    """Compute METEOR score between reference and hypothesis."""
    # Tokenize both hypothesis and reference
    reference_tokens = nltk.word_tokenize(reference.lower())
    hypothesis_tokens = nltk.word_tokenize(hypothesis.lower())
    
    # Compute METEOR score using tokenized inputs
    return single_meteor_score(reference_tokens, hypothesis_tokens)

# ===========================
# EVALUATION FUNCTION
# ===========================
def evaluate(dataloader, num_samples=50):
    """Evaluate caption generation models."""
    all_bleu1 = []
    all_bleu2 = []
    all_bleu3 = []
    all_bleu4 = []
    all_meteor = []
    
    # Process batches until we get enough samples
    total_processed = 0
    for batch in tqdm(dataloader, desc="Evaluating"):
        images = batch['image']
        references = batch['caption']
        
        for i in range(len(images)):
            if total_processed >= num_samples:
                break
                
            image = images[i].to(device)
            reference = references[i]
            
            # Generate caption (returns string)
            hypothesis = generate_caption(image)
            
            # Compute metrics
            b1, b2, b3, b4 = compute_bleu_all(reference, hypothesis)
            meteor = compute_meteor(reference, hypothesis)
            
            all_bleu1.append(b1)
            all_bleu2.append(b2)
            all_bleu3.append(b3)
            all_bleu4.append(b4)
            all_meteor.append(meteor)
            
            # Print progress every 10 samples
            if total_processed % 10 == 0:
                print(f"\nSample {total_processed}:")
                print(f"Reference: {reference}")
                print(f"Generated: {hypothesis}")
                print(f"BLEU-1: {b1:.4f}, BLEU-4: {b4:.4f}, METEOR: {meteor:.4f}")
            
            total_processed += 1
            if total_processed >= num_samples:
                break
    
    # Compute averages
    avg_bleu1 = sum(all_bleu1) / len(all_bleu1)
    avg_bleu2 = sum(all_bleu2) / len(all_bleu2)
    avg_bleu3 = sum(all_bleu3) / len(all_bleu3)
    avg_bleu4 = sum(all_bleu4) / len(all_bleu4)
    avg_meteor = sum(all_meteor) / len(all_meteor)
    
    print("\n=========================")
    print("EVALUATION RESULTS")
    print("=========================")
    print(f"BLEU-1: {avg_bleu1:.4f}")
    print(f"BLEU-2: {avg_bleu2:.4f}")
    print(f"BLEU-3: {avg_bleu3:.4f}")
    print(f"BLEU-4: {avg_bleu4:.4f}")
    print(f"METEOR: {avg_meteor:.4f}")
    
    return {
        "bleu1": avg_bleu1,
        "bleu2": avg_bleu2,
        "bleu3": avg_bleu3,
        "bleu4": avg_bleu4,
        "meteor": avg_meteor
    }

# ===========================
# MAIN EXECUTION
# ===========================
if _name_ == "_main_":
    # Load test dataloader
    loader = get_multicare_dataloader(
        dataset_name='med_test',
        batch_size=8,
        create_new=False,
    )
    
    # Run evaluation
    results = evaluate(loader, num_samples=2)