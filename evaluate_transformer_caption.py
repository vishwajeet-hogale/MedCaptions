import torch
import torch.nn.functional as F
import nltk
import numpy as np
import matplotlib.pyplot as plt
import argparse
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, average_precision_score
from dataloader import get_multicare_dataloader
from deit_encoder import DeiTMedicalEncoder
from transformer_decoder import TransformerMedicalDecoder
from torchvision import transforms
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
from rouge_score import rouge_scorer

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# ===========================
# DEVICE SETUP
# ===========================
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ===========================
# UTILITIES
# ===========================
# Image normalization for visualization
inv_normalize = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
)

def get_caption_embedding(caption, tokenizer, bert_model):
    """Generate BERT CLS embedding for a given caption."""
    inputs = tokenizer(caption, return_tensors='pt', truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :]  # CLS token embedding

def compute_info_nce_loss(embeddings_a, embeddings_b, temperature=0.07):
    """
    Compute InfoNCE contrastive loss for matched image-caption pairs
    
    Args:
        embeddings_a: Predicted caption embeddings from model
        embeddings_b: Ground truth caption embeddings from BERT
        temperature: Temperature parameter for scaling similarity
        
    Returns:
        InfoNCE loss value
    """
    batch_size = embeddings_a.size(0)
    
    # Normalize embeddings
    embeddings_a = F.normalize(embeddings_a, p=2, dim=1)
    embeddings_b = F.normalize(embeddings_b, p=2, dim=1)
    
    # Compute similarities for all pairs
    sim_matrix = torch.matmul(embeddings_a, embeddings_b.T) / temperature
    
    # Labels: diagonal is positive pairs (matching image-caption)
    labels = torch.arange(batch_size).to(device)
    
    # Computing loss in both directions (image→caption and caption→image)
    loss_a = F.cross_entropy(sim_matrix, labels)
    loss_b = F.cross_entropy(sim_matrix.T, labels)
    
    # Average bidirectional loss
    return (loss_a + loss_b) / 2.0

# ===========================
# LOAD MODELS
# ===========================
def load_models(checkpoint_path, tokenizer_path="./MediCareBertTokenizer", bert_model_path="./MediCareBertModel"):
    """Load all required models."""
    # Load BERT tokenizer and model for caption embeddings
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    bert_model = BertModel.from_pretrained(bert_model_path).to(device).eval()
    
    # Load encoder and transformer decoder
    encoder = DeiTMedicalEncoder(embed_size=768)
    decoder = TransformerMedicalDecoder(
        image_embed_size=384,  # Match DeiT encoder output
        freeze_base=False
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load weights
    try:
        decoder.load_state_dict(checkpoint["decoder"])
    except:
        print("Warning: Checkpoint format might be incompatible with transformer decoder")
        
    encoder.load_state_dict(checkpoint["encoder"])
    
    # Move to device and eval mode
    encoder.to(device).eval()
    decoder.to(device).eval()
    
    return tokenizer, bert_model, encoder, decoder

# ===========================
# CAPTION GENERATION
# ===========================
def generate_caption(image_tensor, encoder, decoder, tokenizer, bert_model, reference_loader=None, temperature=0.07):
    """Generate a caption for a given image using nearest neighbor approach."""
    with torch.no_grad():
        # Get image features
        image_feat = encoder(image_tensor.unsqueeze(0).to(device))
        dummy_input = torch.zeros((1, 1, 768)).to(device)
        pred_embedding = decoder(dummy_input, image_feat).squeeze()
        
        # Normalize predicted embedding
        pred_embedding = F.normalize(pred_embedding.unsqueeze(0), p=2, dim=1).squeeze()
        
        # Use the provided reference loader or create a new one
        if reference_loader is None:
            reference_loader = get_multicare_dataloader(
                dataset_name='med_test',
                batch_size=32,
                create_new=False
            )
        
        # Get reference captions from the loader
        all_captions = []
        all_embeddings = []
        
        # Collect all reference captions
        for batch in reference_loader:
            captions = batch['caption']
            all_captions.extend(captions)
            
            # Get caption embeddings from BERT
            with torch.no_grad():
                caption_embeddings = torch.stack([
                    get_caption_embedding(c, tokenizer, bert_model).squeeze() 
                    for c in captions
                ]).to(device)
                # Normalize embeddings
                caption_embeddings = F.normalize(caption_embeddings, p=2, dim=1)
                all_embeddings.append(caption_embeddings)
        
        # Stack all embeddings
        if len(all_embeddings) > 0:
            all_embeddings = torch.cat(all_embeddings, dim=0)
            
            # Find the closest caption using temperature-scaled similarity
            similarities = torch.matmul(pred_embedding.unsqueeze(0), all_embeddings.T) / temperature
            similarities = similarities.squeeze().cpu().numpy()
            
            best_idx = similarities.argmax()
            best_caption = all_captions[best_idx]
            
            return best_caption
        else:
            return "No caption found"

# ===========================
# BLEU COMPUTATION FUNCTIONS
# ===========================
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
# ROUGE COMPUTATION
# ===========================
def compute_rouge(reference, hypothesis):
    """Compute ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference.lower(), hypothesis.lower())
    return scores['rouge1'].fmeasure, scores['rouge2'].fmeasure, scores['rougeL'].fmeasure

# ===========================
# CAPTIONING EVALUATION
# ===========================
def evaluate_captioning(dataloader, encoder, decoder, tokenizer, bert_model, num_samples=50, temperature=0.07):
    """
    Evaluate model using traditional captioning metrics (BLEU, ROUGE, METEOR)
    """
    encoder.eval()
    decoder.eval()
    
    all_bleu1 = []
    all_bleu2 = []
    all_bleu3 = []
    all_bleu4 = []
    all_meteor = []
    all_rouge1 = []
    all_rouge2 = []
    all_rougeL = []
    
    # Create a reference loader for caption generation
    reference_loader = get_multicare_dataloader(
        dataset_name='med_test',
        batch_size=32,
        create_new=False
    )
    
    # Process batches until we get enough samples
    total_processed = 0
    examples = []
    
    for batch in tqdm(dataloader, desc="Evaluating captioning metrics"):
        images = batch['image']
        references = batch['caption']
        
        for i in range(len(images)):
            if total_processed >= num_samples:
                break
                
            image = images[i].to(device)
            reference = references[i]
            
            # Generate caption
            hypothesis = generate_caption(
                image, encoder, decoder, tokenizer, bert_model, 
                reference_loader=reference_loader, temperature=temperature
            )
            
            # Compute metrics
            b1, b2, b3, b4 = compute_bleu_all(reference, hypothesis)
            meteor = compute_meteor(reference, hypothesis)
            r1, r2, rL = compute_rouge(reference, hypothesis)
            
            all_bleu1.append(b1)
            all_bleu2.append(b2)
            all_bleu3.append(b3)
            all_bleu4.append(b4)
            all_meteor.append(meteor)
            all_rouge1.append(r1)
            all_rouge2.append(r2)
            all_rougeL.append(rL)
            
            # Save example for display
            examples.append({
                'image': image.cpu(),
                'reference': reference,
                'hypothesis': hypothesis,
                'metrics': {
                    'bleu1': b1,
                    'bleu4': b4,
                    'meteor': meteor,
                    'rouge1': r1
                }
            })
            
            # Print progress every 10 samples
            if total_processed % 10 == 0:
                print(f"\nSample {total_processed}:")
                print(f"Reference: {reference}")
                print(f"Generated: {hypothesis}")
                print(f"BLEU-1: {b1:.4f}, BLEU-4: {b4:.4f}, METEOR: {meteor:.4f}, ROUGE-L: {rL:.4f}")
            
            total_processed += 1
            if total_processed >= num_samples:
                break
    
    # Compute averages
    avg_bleu1 = sum(all_bleu1) / len(all_bleu1) if all_bleu1 else 0
    avg_bleu2 = sum(all_bleu2) / len(all_bleu2) if all_bleu2 else 0
    avg_bleu3 = sum(all_bleu3) / len(all_bleu3) if all_bleu3 else 0
    avg_bleu4 = sum(all_bleu4) / len(all_bleu4) if all_bleu4 else 0
    avg_meteor = sum(all_meteor) / len(all_meteor) if all_meteor else 0
    avg_rouge1 = sum(all_rouge1) / len(all_rouge1) if all_rouge1 else 0
    avg_rouge2 = sum(all_rouge2) / len(all_rouge2) if all_rouge2 else 0
    avg_rougeL = sum(all_rougeL) / len(all_rougeL) if all_rougeL else 0
    
    # Print results
    print("\n==========================")
    print("CAPTIONING METRICS")
    print("==========================")
    print(f"BLEU-1: {avg_bleu1:.4f}")
    print(f"BLEU-2: {avg_bleu2:.4f}")
    print(f"BLEU-3: {avg_bleu3:.4f}")
    print(f"BLEU-4: {avg_bleu4:.4f}")
    print(f"METEOR: {avg_meteor:.4f}")
    print(f"ROUGE-1: {avg_rouge1:.4f}")
    print(f"ROUGE-2: {avg_rouge2:.4f}")
    print(f"ROUGE-L: {avg_rougeL:.4f}")
    
    # Visualize a few examples
    for i, example in enumerate(examples[:5]):  # Show first 5 examples
        img_np = inv_normalize(example['image']).permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)
        
        plt.figure(figsize=(10, 6))
        plt.imshow(img_np)
        plt.title(f"Example {i+1}")
        plt.axis("off")
        
        plt.figtext(0.5, 0.01, f"Reference: {example['reference']}\n\n" + 
                             f"Generated: {example['hypothesis']}\n\n" +
                             f"BLEU-1: {example['metrics']['bleu1']:.4f}, " +
                             f"BLEU-4: {example['metrics']['bleu4']:.4f}, " + 
                             f"METEOR: {example['metrics']['meteor']:.4f}, " +
                             f"ROUGE-1: {example['metrics']['rouge1']:.4f}", 
                   ha="center", fontsize=10, wrap=True)
        
        plt.tight_layout()
        plt.savefig(f'caption_example_{i+1}.png', bbox_inches='tight')
        plt.close()
    
    return {
        "bleu1": avg_bleu1,
        "bleu2": avg_bleu2,
        "bleu3": avg_bleu3,
        "bleu4": avg_bleu4,
        "meteor": avg_meteor,
        "rouge1": avg_rouge1,
        "rouge2": avg_rouge2,
        "rougeL": avg_rougeL
    }

# ===========================
# CONTRASTIVE EVALUATION
# ===========================
def evaluate_contrastive(dataloader, encoder, decoder, tokenizer, bert_model, temperature=0.07):
    """
    Evaluate model using contrastive metrics
    """
    encoder.eval()
    decoder.eval()
    
    val_loss = 0.0
    val_steps = 0
    
    all_similarities = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="Evaluating contrastive metrics"):
        images = batch['image'].to(device)
        captions = batch['caption']
        batch_size = images.size(0)
        
        with torch.no_grad():
            # Get image features
            image_features = encoder(images)
            
            # Get caption embeddings
            caption_embeddings = torch.stack([
                get_caption_embedding(c, tokenizer, bert_model).squeeze() 
                for c in captions
            ]).to(device)
            
            # Forward through decoder
            dummy_input = torch.zeros((batch_size, 1, 768)).to(device)
            pred_embeddings = decoder(dummy_input, image_features)
            
            # Compute loss
            loss = compute_info_nce_loss(
                pred_embeddings, 
                caption_embeddings,
                temperature=temperature
            )
            
            # For computing similarity distribution
            pred_norm = F.normalize(pred_embeddings, p=2, dim=1)
            caption_norm = F.normalize(caption_embeddings, p=2, dim=1)
            
            # Compute similarity matrix
            sim_matrix = torch.matmul(pred_norm, caption_norm.T) / temperature
            
            # Gather positive and negative similarities
            for i in range(batch_size):
                for j in range(batch_size):
                    sim = sim_matrix[i, j].item()
                    # 1 for positive pairs, 0 for negative pairs
                    label = 1 if i == j else 0
                    
                    all_similarities.append(sim)
                    all_labels.append(label)
            
            val_loss += loss.item()
            val_steps += 1
    
    # Calculate average loss
    avg_loss = val_loss / val_steps if val_steps > 0 else float('inf')
    
    # Convert to numpy for metrics calculation
    all_similarities = np.array(all_similarities)
    all_labels = np.array(all_labels)
    
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(all_labels, all_similarities)
    ap_score = average_precision_score(all_labels, all_similarities)
    
    # Print results
    print("\n==========================")
    print("CONTRASTIVE METRICS")
    print("==========================")
    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"Average Precision: {ap_score:.4f}")
    
    # Plot precision-recall curve
    plt.figure(figsize=(10, 7))
    plt.plot(recall, precision, marker='.', label=f'AP={ap_score:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('transformer_precision_recall.png')
    plt.close()
    
    # Plot similarity distributions
    plt.figure(figsize=(10, 7))
    pos_sims = all_similarities[all_labels == 1]
    neg_sims = all_similarities[all_labels == 0]
    plt.hist(pos_sims, bins=50, alpha=0.5, label='Positive Pairs')
    plt.hist(neg_sims, bins=50, alpha=0.5, label='Negative Pairs')
    plt.xlabel('Similarity')
    plt.ylabel('Count')
    plt.title('Similarity Distribution')
    plt.legend()
    plt.grid(True)
    plt.savefig('transformer_similarity_dist.png')
    plt.close()
    
    return {
        "validation_loss": avg_loss,
        "average_precision": ap_score
    }

# ===========================
# QUALITATIVE EVALUATION
# ===========================
def qualitative_evaluation(dataloader, encoder, decoder, tokenizer, bert_model, num_samples=5, temperature=0.07):
    """
    Show qualitative evaluation examples with most similar captions
    """
    batch = next(iter(dataloader))
    images = batch['image'].to(device)
    captions = batch['caption']
    
    print(f"\nQualitative evaluation with {len(images)} images...")
    
    # Encode ground truth captions
    with torch.no_grad():
        caption_embeddings = torch.stack([
            get_caption_embedding(c, tokenizer, bert_model).squeeze() for c in captions
        ]).to(device)
        # Normalize caption embeddings
        caption_embeddings = F.normalize(caption_embeddings, p=2, dim=1)
    
    # Process samples
    for i in range(min(num_samples, len(images))):
        with torch.no_grad():
            # Get image features
            image_feat = encoder(images[i].unsqueeze(0))
            dummy_input = torch.zeros((1, 1, 768)).to(device)
            pred_embed = decoder(dummy_input, image_feat).squeeze()
            # Normalize predicted embedding
            pred_embed = F.normalize(pred_embed.unsqueeze(0), p=2, dim=1).squeeze()
            
            # Calculate similarities
            sims = torch.matmul(pred_embed.unsqueeze(0), caption_embeddings.T) / temperature
            sims = sims.squeeze().cpu().numpy()
            
            # Get top matches
            top_indices = sims.argsort()[-3:][::-1]
            
            # Display results
            print(f"\nImage {i+1}")
            print(f"Ground Truth: {captions[i]}")
            print(f"Top 3 Retrieved Captions:")
            for j, idx in enumerate(top_indices):
                print(f"   {j+1}. {captions[idx]} (sim: {sims[idx]:.4f})")
            
            # Show image
            img_np = inv_normalize(images[i].cpu()).permute(1, 2, 0).numpy()
            img_np = np.clip(img_np, 0, 1)
            plt.figure(figsize=(10, 7))
            plt.imshow(img_np)
            plt.title(f"Sample {i+1}")
            plt.axis("off")
            plt.savefig(f'transformer_sample_{i+1}.png')
            plt.close()

# ===========================
# MAIN FUNCTION
# ===========================
def main(args):
    print("\nLoading models...")
    tokenizer, bert_model, encoder, decoder = load_models(args.checkpoint)
    
    # Define dataset filters
    if args.filter:
        # Example filters
        filters = [
            {'field': 'label', 'string_list': ['mri', 'head']},
            {'field': 'caption', 'string_list': ['showing', 'demonstrates', 'reveals'], 'operator': 'any'}
        ]
    else:
        filters = None
    
    # Create dataloader
    dataloader = get_multicare_dataloader(
        dataset_name=args.split,
        batch_size=args.batch_size,
        create_new=False,
        filters=filters
    )
    
    # Run evaluations
    if args.qualitative:
        print("\nRunning qualitative evaluation...")
        qualitative_evaluation(
            dataloader, encoder, decoder, tokenizer, bert_model, 
            num_samples=args.num_samples, temperature=args.temperature
        )
    
    if args.captioning:
        print("\nRunning captioning evaluation (BLEU, ROUGE, METEOR)...")
        captioning_results = evaluate_captioning(
            dataloader, encoder, decoder, tokenizer, bert_model, 
            num_samples=args.num_samples, temperature=args.temperature
        )
    
    if args.contrastive:
        print("\nRunning contrastive evaluation...")
        contrastive_results = evaluate_contrastive(
            dataloader, encoder, decoder, tokenizer, bert_model, 
            temperature=args.temperature
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate transformer-based medical image captioning with InfoNCE')
    parser.add_argument('--checkpoint', type=str, default="checkpoints/checkpoint_epoch_25.pt",
                        help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default="med_test",
                        help='Dataset split to use: med_train, med_val, med_test')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for evaluation')
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='Temperature for InfoNCE similarity scaling')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples for qualitative evaluation')
    parser.add_argument('--filter', action='store_true',
                        help='Apply MRI head filters to the dataset')
    parser.add_argument('--qualitative', action='store_true',
                        help='Run qualitative evaluation')
    parser.add_argument('--captioning', action='store_true', 
                        help='Run captioning metrics evaluation (BLEU, ROUGE, METEOR)')
    parser.add_argument('--contrastive', action='store_true',
                        help='Run contrastive metrics evaluation')
    parser.add_argument('--all', action='store_true',
                        help='Run all evaluation modes')
    
    args = parser.parse_args()
    
    # If --all is passed, enable all evaluation modes
    if args.all:
        args.qualitative = True
        args.captioning = True 
        args.contrastive = True
    
    # If no modes specified, default to captioning metrics
    if not (args.qualitative or args.captioning or args.contrastive):
        args.captioning = True
    
    main(args) 