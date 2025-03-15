import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import random
import os
import json
from collections import defaultdict

nltk.download('wordnet')  # Required for METEOR

def generate_synthetic_references(original_caption, num_variants=3):
    """
    Generate alternative captions using simple word replacements.
    
    Args:
        original_caption (str): The original caption for the image.
        num_variants (int): Number of synthetic reference captions to generate.
        
    Returns:
        list: A list of alternative captions.
    """
    synonym_dict = {
        "MRI": ["magnetic resonance imaging", "scan", "MRI scan"],
        "tumor": ["mass", "abnormality", "growth"],
        "brain": ["head", "cerebrum"],
        "showing": ["depicting", "revealing"],
        "image": ["picture", "photo", "scan"],
        "of": ["with", "containing"],
        "a": ["one", "an"],
        "is": ["shows", "demonstrates"]
    }
    
    words = original_caption.split()
    alternative_captions = [original_caption]  # Keep the original caption

    for _ in range(num_variants):  # Generate multiple variations
        new_caption = [
            random.choice(synonym_dict.get(word, [word])) for word in words
        ]
        alternative_captions.append(" ".join(new_caption))
    
    return alternative_captions


def build_reference_caption_dict_from_dataloader(dataloader):
    """
    Build a dictionary mapping image_id to generated reference captions from the test DataLoader.

    Args:
        dataloader (DataLoader): PyTorch DataLoader for test dataset.

    Returns:
        dict: Dictionary where keys are image_ids and values are lists of generated reference captions.
    """
    reference_dict = {}

    for batch in dataloader:  # Iterate through the test set
        image_ids = batch['image_id']  # List of image IDs
        captions = batch['caption']    # Corresponding captions

        for image_id, caption in zip(image_ids, captions):
            reference_dict[image_id] = generate_synthetic_references(caption)  # Generate reference captions

    return reference_dict
def evaluate_bleu(image_id, generated_caption, reference_caption_dict, n_gram=4):
    """
    Compute BLEU score for a generated caption using multiple reference captions.

    Args:
        image_id (str): Image ID.
        generated_caption (str): Model-generated caption.
        reference_caption_dict (dict): Dictionary of image_id → list of generated reference captions.
        n_gram (int): Max n-gram to consider (1 for BLEU-1, 4 for BLEU-4).

    Returns:
        float: BLEU score (0 to 1).
    """
    references = reference_caption_dict.get(image_id, [])  # Get reference captions
    if not references:
        print(f"Warning: No reference captions found for image_id {image_id}")
        return 0.0  # Return 0 if no references exist

    references_tokenized = [ref.split() for ref in references]  # Tokenize reference captions
    hypothesis = generated_caption.split()  # Tokenize generated caption

    smoothing = SmoothingFunction().method1  # Helps with short sentences

    return sentence_bleu(references_tokenized, hypothesis, weights=[1/n_gram]*n_gram, smoothing_function=smoothing)
def evaluate_model_bleu(test_dataloader, model):
    """
    Evaluate a model's captioning performance on the test set using BLEU scores.

    Args:
        test_dataloader (DataLoader): PyTorch DataLoader for test dataset.
        model: The captioning model to evaluate.

    Returns:
        dict: Dictionary where keys are image_ids and values are BLEU scores.
    """
    # Step 1: Generate reference captions for the test dataset
    reference_caption_dict = build_reference_caption_dict_from_dataloader(test_dataloader)

    # Step 2: Generate captions using the model
    bleu_scores = {}

    for batch in test_dataloader:
        images = batch['image']  # Extract images
        image_ids = batch['image_id']  # Extract image IDs

        # Generate captions using the model
        generated_captions = model.generate_caption(images)  # Assuming model has a `generate_caption` method

        # Step 3: Compute BLEU scores
        for i, image_id in enumerate(image_ids):
            bleu_score = evaluate_bleu(image_id, generated_captions[i], reference_caption_dict)
            bleu_scores[image_id] = bleu_score  # Store BLEU score per image

    return bleu_scores

def evaluate_meteor(image_id, generated_caption, reference_caption_dict):
    """
    Compute METEOR score for a generated caption using multiple reference captions.

    Args:
        image_id (str): Image ID.
        generated_caption (str): Model-generated caption.
        reference_caption_dict (dict): Dictionary of image_id → list of reference captions.

    Returns:
        float: METEOR score (0 to 1).
    """
    references = reference_caption_dict.get(image_id, [])  # Get reference captions
    if not references:
        print(f"Warning: No reference captions found for image_id {image_id}")
        return 0.0  # Return 0 if no references exist

    references_tokenized = [ref.split() for ref in references]  # Tokenized reference captions
    hypothesis = generated_caption.split()  # Tokenized generated caption

    # Compute METEOR score using NLTK's implementation
    return meteor_score(references_tokenized, hypothesis)


if __name__ == "__main__":


    # print(f"BLEU Score: {bleu_score:.4f}")
    # print(f"METEOR Score: {meteor_score_val:.4f}")
    
    # # Assuming `test_dataloader` is already defined
    # # Assuming `model` has a `.generate_caption(images)` method

    # bleu_results = evaluate_model_bleu(test_dataloader, model)

    # # Print BLEU scores for the first few images
    # for image_id, score in list(bleu_results.items())[:5]:
    #     print(f"Image ID: {image_id}, BLEU Score: {score:.4f}")
    pass