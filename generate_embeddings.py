import torch
from transformers import BertTokenizer, BertModel

def load_embedding_model(model_path="MediCareBert"):
    """
    Load the model and tokenizer for generating embeddings.
    """
    tokenizer = BertTokenizer.from_pretrained(model_path)
    # Use BertModel so we can easily access hidden states/pooler outputs
    model = BertModel.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

def get_caption_embedding(caption, tokenizer, model, device=torch.device("cpu")):
    """
    Given a single caption (string), generate the embedding using the [CLS] token.
    """
    # Tokenize the input
    inputs = tokenizer(caption, return_tensors='pt', truncation=True, max_length=256)
    # Move to the chosen device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        # Option 1: Use last_hidden_state (batch_size x seq_len x hidden_size)
        last_hidden_state = outputs.last_hidden_state
        # The [CLS] token embedding is at index 0
        cls_embedding = last_hidden_state[:, 0, :]  # shape: (batch_size, hidden_size)

        # Option 2: Use pooler_output (if your model includes a pooled output)
        # pooled_output = outputs.pooler_output  # shape: (batch_size, hidden_size)

    # Return the CLS embedding as a CPU tensor (batch_size=1 for single input)
    return cls_embedding.squeeze().cpu()

if __name__ == "__main__":
    # Load your fine-tuned model
    tokenizer, model = load_embedding_model("MediCareBert")

    # Test on a single caption
    test_caption = "MRI scan reveals a small lesion in the left temporal lobe."
    embedding = get_caption_embedding(test_caption, tokenizer, model)
    print("Caption embedding shape:", embedding.shape)
    print("Caption embedding:", embedding)
