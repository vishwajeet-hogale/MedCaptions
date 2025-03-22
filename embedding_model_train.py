from transformers import BertTokenizer, BertForMaskedLM, AdamW
import torch
from tqdm import tqdm
from dataloader import get_multicare_dataloader

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")

def load_dataloader():
    caption_filters = [
        {'field': 'label', 'string_list': ['mri', 'head']},
        {'field': 'caption', 'string_list': ['showing', 'demonstrates', 'reveals'], 'operator': 'any'}
    ]
    return get_multicare_dataloader(
        dataset_name='brain_tumor_mri_1742661954',
        batch_size=8,
        create_new=True,
        filters=caption_filters
    )

def convert_dataloader_captions_to_list(loader):
    print("Extracting captions from the dataloader!")
    all_captions = []
    for batch in loader:
        # batch should be a dict with key 'caption'
        # adjust if your real dataloader yields something different
        all_captions.extend(batch['caption'])
    return all_captions

def tokenize_captions(all_captions):
    # Tokenize all captions into a single batch encoding
    inputs = tokenizer(
        all_captions,
        return_tensors='pt',
        max_length=256,
        truncation=True,
        padding='max_length'
    )
    # Clone input_ids to labels
    inputs['labels'] = inputs['input_ids'].clone()
    return inputs

def create_masks_captions(tokenized_inputs):
    # tokenized_inputs is a BatchEncoding, but can behave like a dict
    input_ids = tokenized_inputs['input_ids']
    rand = torch.rand(input_ids.shape)

    # Create mask array of booleans: 15% chance, excluding special tokens
    #   101 is [CLS], 0 is [PAD], etc.
    mask_arr = (rand < 0.15) & (input_ids != 101) & (input_ids != 0)

    for i in range(mask_arr.shape[0]):
        # Mask selected tokens with the [MASK] ID = 103
        input_ids[i, mask_arr[i]] = 103

    tokenized_inputs['input_ids'] = input_ids
    return tokenized_inputs

class MediCareBertDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        # Convert each field’s element at 'idx' into a tensor
        return {
            key: torch.tensor(val[idx])
            for key, val in self.encodings.items()
        }

    def __len__(self):
        # Number of samples is the first dimension of 'input_ids'
        return self.encodings['input_ids'].shape[0]

def train(model, dataloader, epochs=1):
    # Pick a device (mps for M1, or cuda/cpu as needed)
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=1e-5)

    for epoch in range(epochs):
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch+1} | Loss = {loss.item():.4f}")
    model.save_pretrained("MediCareBert")
    tokenizer.save_pretrained("MediCareBert")    

if __name__ == "__main__":
    # 1. Get your custom loader
    loader = load_dataloader()

    # 2. Collect all captions into a list
    all_captions = convert_dataloader_captions_to_list(loader)

    # 3. Tokenize them
    inputs = tokenize_captions(all_captions)

    # 4. Randomly mask 15% of tokens
    inputs = create_masks_captions(inputs)

    # 5. Create a Dataset and DataLoader for training
    dataset = MediCareBertDataset(inputs)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    # 6. Train the model
    train(model, dataloader, epochs=1)

