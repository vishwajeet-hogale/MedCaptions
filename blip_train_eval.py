import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
# Assuming multiversity.multicare_dataset is correctly installed or available
from multiversity.multicare_dataset import MedicalDatasetCreator
import json
from transformers import BlipProcessor, BlipForConditionalGeneration
from torch.cuda.amp import autocast, GradScaler
import evaluate




class MultiCaReDataset(Dataset):
    """Dataset class for loading MultiCaRe medical images and captions for image captioning."""

    def __init__(self, dataset_directory='medical_datasets', dataset_name='brain_tumor_mri',
                 processor=None, create_dataset=False, filters=None,
                 data_subset_indices=None): # Added data_subset_indices
        """
        Initialize the MultiCaRe dataset for image captioning.

        Args:
            dataset_directory (str): Directory where the MultiCaRe data is stored
            dataset_name (str): Name of the specific dataset
            processor (BlipProcessor, optional): BLIP processor for image and text tokenization
            create_dataset (bool): Whether to create a new dataset
            filters (list): Filters to use when creating a new dataset
            data_subset_indices (list, optional): A list of indices to use if this dataset
                                                  is a subset (e.g., train/test split).
        """
        self.dataset_directory = dataset_directory
        self.dataset_name = dataset_name
        self.processor = processor

        # Create dataset if requested
        if create_dataset:
            if not filters:
                # Default filters to get MRI images with captions
                filters = [
                    {'field': 'label', 'string_list': ['mri']},
                    {'field': 'caption', 'string_list': [''], 'operator': 'any'}  # Any image with caption
                ]
            # Ensure create_multicare_dataset is called if a new dataset is needed
            _ = create_multicare_dataset(dataset_name, filters, dataset_directory)

        # Load the dataset metadata
        all_data = self._load_metadata()

        # Filter out entries without captions
        captioned_data = [item for item in all_data if item.get('caption', '').strip()]
        print(f"Filtered to {len(captioned_data)} samples with captions")

        # Filter out entries with missing images
        self.full_data = self._filter_valid_images(captioned_data) # Store full valid data

        # Apply data_subset_indices if provided
        if data_subset_indices is not None:
            self.data = [self.full_data[i] for i in data_subset_indices]
            print(f"Loaded {len(self.data)} samples for subset from {dataset_name} dataset")
        else:
            self.data = self.full_data # If no subset, use all valid data
            print(f"Loaded {len(self.data)} samples with valid images and captions from {dataset_name} dataset")


    def _load_metadata(self):
        """Load the dataset metadata from JSON file."""
        try:
            # Construct path to the metadata JSON file (try both possible filenames)
            metadata_path = os.path.join(self.dataset_directory,
                                        self.dataset_name,
                                        'image_metadata.json')

            if not os.path.exists(metadata_path):
                metadata_path = os.path.join(self.dataset_directory,
                                            self.dataset_name,
                                            'images_metadata.json')

            if not os.path.exists(metadata_path):
                raise FileNotFoundError(f"Could not find metadata file for dataset {self.dataset_name}")

            print(f"Loading metadata from: {metadata_path}")

            # Read and parse the JSON file
            data = []
            with open(metadata_path, 'r') as f:
                for line in f:
                    if line.strip():  # Skip empty lines
                        data.append(json.loads(line))

            print(f"Loaded {len(data)} total entries from metadata file")
            return data

        except Exception as e:
            print(f"Error loading dataset: {e}")
            return []

    def _get_image_path(self, sample_data):
        """Determine the correct image path for a sample."""
        # Try different ways to get the image path
        if 'file' in sample_data:
            # Most common in the demo
            path = os.path.join(self.dataset_directory,
                               self.dataset_name,
                               'images',
                               sample_data['file'])
            if os.path.exists(path):
                return path

        # Try file_path
        if 'file_path' in sample_data:
            # Try different ways to interpret file_path
            possible_paths = [
                # Full path as provided
                os.path.join(self.dataset_directory, sample_data['file_path']),
                # Just the filename in images directory
                os.path.join(self.dataset_directory, self.dataset_name, 'images', os.path.basename(sample_data['file_path'])),
                # Direct path
                sample_data['file_path']
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    return path

        # Try to construct one from main_image or file_id
        for id_field in ['main_image', 'file_id']:
            if id_field in sample_data:
                file_id = sample_data[id_field]

                # Try common patterns for file paths
                possible_paths = [
                    os.path.join(self.dataset_directory, self.dataset_name, 'images', file_id),
                    os.path.join(self.dataset_directory, self.dataset_name, 'images', f"{file_id}.jpg"),
                    os.path.join(self.dataset_directory, self.dataset_name, 'images', f"{file_id}.png"),
                    os.path.join(self.dataset_directory, self.dataset_name, 'images', f"{file_id}.webp"),
                    os.path.join(self.dataset_directory, 'whole_multicare_dataset', 'images', file_id),
                    os.path.join(self.dataset_directory, 'images', file_id),
                    # Try with PMC folder if it starts with PMC
                    os.path.join(self.dataset_directory, 'whole_multicare_dataset', f"PMC{file_id.split('_')[0]}" if '_' in file_id and file_id.split('_')[0].isdigit() else "", file_id)
                ]

                for path in possible_paths:
                    if os.path.exists(path):
                        return path

        # No valid path found
        return None

    def _filter_valid_images(self, data):
        """Filter the dataset to only include samples with valid images."""
        valid_data = []
        missing_count = 0

        for sample in data:
            image_path = self._get_image_path(sample)
            if image_path is not None:
                # Add the resolved path to the sample for faster access later
                sample['_resolved_image_path'] = image_path
                valid_data.append(sample)
            else:
                missing_count += 1

        if missing_count > 0:
            print(f"Excluded {missing_count} samples with missing images")

        return valid_data

    def __len__(self):
        """Return the total number of samples."""
        return len(self.data)

    def __getitem__(self, idx):
        """Return a sample (image, caption) from the dataset."""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_data = self.data[idx]
        image_path = sample_data['_resolved_image_path']

        try:
            image = Image.open(image_path).convert('RGB')
            caption = sample_data.get('caption', '')

            if self.processor:
                # Truncate caption tokens to 512
                encoding = self.processor(images=image, text=caption, return_tensors="pt", padding='max_length', max_length=512, truncation=True)
                encoding = {k: v.squeeze() for k, v in encoding.items()}
                encoding['caption'] = caption[:512]  # Optionally truncate raw caption string for reference/debug

                return encoding

            else:
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                image = transform(image)

                return {
                    'image': image,
                    'caption': caption[:512],  # Truncate raw caption string
                    'image_id': sample_data.get('file_id', sample_data.get('main_image', f"unknown_{idx}"))
                }

        except Exception as e:
            print(f"Unexpected error loading image {image_path}: {e}")
            raise


def create_multicare_dataset(dataset_name, filters=None, dataset_directory='medical_datasets'):
    """
    Create a new MultiCaRe dataset subset if it doesn't already exist.

    Args:
        dataset_name (str): Name of the dataset to create
        filters (list): List of filter dictionaries to apply
        dataset_directory (str): Directory to store the dataset

    Returns:
        MedicalDatasetCreator: The initialized creator object
    """
    # Check if the dataset already exists
    dataset_dir = os.path.join(dataset_directory, dataset_name)
    metadata_path = os.path.join(dataset_dir, 'image_metadata.json')
    # Also check the alternative name
    if not os.path.exists(metadata_path):
        metadata_path = os.path.join(dataset_dir, 'images_metadata.json')

    if os.path.exists(metadata_path):
        print(f"Dataset '{dataset_name}' metadata file already exists at '{metadata_path}', skipping creation.")
        # Return a dummy creator if it exists, as its purpose is for creation.
        return MedicalDatasetCreator(directory=dataset_directory)


    # Initialize MedicalDatasetCreator
    print(f"Initializing MedicalDatasetCreator to create dataset: {dataset_name}")
    mdc = MedicalDatasetCreator(directory=dataset_directory)

    # Create new dataset with filters
    if filters:
        mdc.create_dataset(
            dataset_name=dataset_name,
            filter_list=filters,
            dataset_type='multimodal'  # Need multimodal for both images and captions
        )
        print(f"Created new dataset: {dataset_name}")
    else:
        raise ValueError("Filters must be provided to create a new dataset when create_dataset is True")

    return mdc


def get_multicare_dataloaders_with_split(
    dataset_name, processor, batch_size=16, num_workers=4,
    dataset_directory='medical_datasets', create_new=False, filters=None,
    train_ratio=0.8, seed=42):
    """
    Create train and test DataLoaders for a MultiCaRe dataset, with a train-test split.

    Args:
        dataset_name (str): Name of the dataset to use.
        processor (BlipProcessor): BLIP processor for image and text tokenization.
        batch_size (int): Batch size for training.
        num_workers (int): Number of worker threads for loading data.
        dataset_directory (str): Directory where MultiCaRe data is stored.
        create_new (bool): Whether to create a new dataset if it doesn't exist.
        filters (list): Filters to use when creating a new dataset.
        train_ratio (float): The proportion of the dataset to use for training (e.g., 0.8 for 80/20 split).
        seed (int): Seed for reproducible train-test split.

    Returns:
        tuple: (train_dataloader, test_dataloader)
    """
    # Determine if we need to create the dataset
    dataset_dir = os.path.join(dataset_directory, dataset_name)
    metadata_path = os.path.join(dataset_dir, 'image_metadata.json')
    if not os.path.exists(metadata_path): # Check for main metadata file
        metadata_path = os.path.join(dataset_dir, 'images_metadata.json') # Check for alternative

    # Only set create_dataset to True if create_new is True AND the metadata file does NOT exist
    actually_create_dataset = create_new and not os.path.exists(metadata_path)

    # Load the full dataset first (without applying any subset indices yet)
    # The MultiCaReDataset will load all valid images and captions from the specified dataset_name.
    full_dataset = MultiCaReDataset(
        dataset_directory=dataset_directory,
        dataset_name=dataset_name,
        processor=processor,
        create_dataset=actually_create_dataset, # Only create if needed
        filters=filters,
        data_subset_indices=None # Load the full set first
    )

    # Check if dataset is empty
    if len(full_dataset) == 0:
        raise ValueError(f"Dataset '{dataset_name}' has 0 valid samples after filtering. "
                         "Check image paths, filters, and make sure images are accessible.")

    # Perform train-test split
    dataset_size = len(full_dataset)
    train_size = int(train_ratio * dataset_size)
    test_size = dataset_size - train_size

    # Ensure reproducibility of the split
    generator = torch.Generator().manual_seed(seed)
    train_indices, test_indices = random_split(
        range(dataset_size), [train_size, test_size], generator=generator
    )

    print(f"Splitting dataset into {len(train_indices)} training samples and {len(test_indices)} testing samples.")

    # Create separate dataset objects for train and test, passing their respective indices
    train_dataset = MultiCaReDataset(
        dataset_directory=dataset_directory,
        dataset_name=dataset_name,
        processor=processor,
        create_dataset=False, # Already created/loaded by full_dataset
        filters=None, # Filters already applied when creating full_dataset
        data_subset_indices=train_indices.indices # Pass the actual indices
    )

    test_dataset = MultiCaReDataset(
        dataset_directory=dataset_directory,
        dataset_name=dataset_name,
        processor=processor,
        create_dataset=False, # Already created/loaded by full_dataset
        filters=None,
        data_subset_indices=test_indices.indices # Pass the actual indices
    )

    # Define the custom collate function here, inside get_multicare_dataloaders_with_split
    # This allows it to access the 'processor' argument from the outer function's scope.
    def custom_collate_fn(batch):
        """Custom collate function for BLIP training."""
        pixel_values = [item["pixel_values"] for item in batch]
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        captions = [item["caption"] for item in batch]

        pixel_values_tensor = torch.stack(pixel_values)

        # FIX: Remove 'truncation', set max_length=512 for BLIP compatibility
        tokenized_batch = processor.tokenizer.pad(
            {'input_ids': input_ids, 'attention_mask': attention_mask},
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )

        return {
            "pixel_values": pixel_values_tensor,
            "input_ids": tokenized_batch["input_ids"],
            "attention_mask": tokenized_batch["attention_mask"],
            "captions": captions 
        }

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True, # Always shuffle training data
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn # Use the new custom_collate_fn
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False, # No need to shuffle test data
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn # Use the new custom_collate_fn
    )

    return train_dataloader, test_dataloader

# def train_blip_model(model, dataloader, num_epochs=3, learning_rate=5e-5, device="cuda"):
#     """
#     Training loop for BLIP model on image captioning task.
#     """
#     model.to(device)
#     optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
#     scaler = GradScaler()
#     model.train()
#     print(f"\nStarting training for {num_epochs} epochs...")
#     for epoch in range(num_epochs):
#         total_loss = 0
#         for i, batch in enumerate(dataloader):
#             pixel_values = batch["pixel_values"].to(device)
#             input_ids = batch["input_ids"].to(device)
#             attention_mask = batch["attention_mask"].to(device)
#             with autocast():
#                 outputs = model(
#                     pixel_values=pixel_values,
#                     input_ids=input_ids,
#                     attention_mask=attention_mask,
#                     labels=input_ids
#                 )
#                 loss = outputs.loss
#             total_loss += loss.item()
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
#             optimizer.zero_grad()
#             if (i + 1) % 10 == 0:
#                 print(f"Epoch {epoch+1}/{num_epochs}, Step {i+1}/{len(dataloader)}, Loss: {loss.item():.4f}")
#         print(f"Epoch {epoch+1} finished. Average Loss: {total_loss / len(dataloader):.4f}")
#     print("Training complete.")


def train_blip_model(model, dataloader, num_epochs=3, learning_rate=5e-5, device="cuda", grad_accum_steps=4):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = GradScaler()
    model.train()
    print(f"\nStarting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        total_loss = 0
        optimizer.zero_grad()
        for i, batch in enumerate(dataloader):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with autocast():
                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
                loss = outputs.loss / grad_accum_steps
            total_loss += loss.item() * grad_accum_steps
            scaler.scale(loss).backward()
            if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(dataloader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            if (i + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Step {i+1}/{len(dataloader)}, Loss: {loss.item() * grad_accum_steps:.4f}")
        print(f"Epoch {epoch+1} finished. Average Loss: {total_loss / len(dataloader):.4f}")
    print("Training complete.")

def evaluate_blip_model(model, dataloader, processor, device="cuda"):
    """
    Evaluates BLIP model on test set using BLEU, METEOR, and ROUGE-L.
    """
    model.eval()
    model.to(device)
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")
    generated_captions = []
    reference_captions = []
    print("\nStarting evaluation...")
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            pixel_values = batch["pixel_values"].to(device)
            batch_reference_captions = batch["captions"]
            with autocast():
                outputs = model.generate(
                    pixel_values=pixel_values,
                    max_new_tokens=50,
                    num_beams=4,
                    early_stopping=True
                )
            decoded_captions = processor.batch_decode(outputs, skip_special_tokens=True)
            print(decoded_captions)
            generated_captions.extend(decoded_captions)
            reference_captions.extend(batch_reference_captions)
            if (i + 1) % 10 == 0:
                print(f"Processed {len(generated_captions)} samples for evaluation.")
    bleu_results = bleu.compute(predictions=generated_captions, references=[[ref] for ref in reference_captions])
    rouge_results = rouge.compute(predictions=generated_captions, references=reference_captions)
    meteor_results = meteor.compute(predictions=generated_captions, references=reference_captions)
    print("\n--- Evaluation Results ---")
    print(f"BLEU: {bleu_results['bleu']:.4f}")
    print(f"ROUGE-L: {rouge_results['rougeL']:.4f}")
    print(f"METEOR: {meteor_results['meteor']:.4f}")
    print("------------------------")
    return {
        "bleu": bleu_results['bleu'],
        "rougeL": rouge_results['rougeL'],
        "meteor": meteor_results['meteor']
    }

def main():
    MODEL_NAME = "Salesforce/blip-image-captioning-base"
    DATASET_DIR = 'medical_datasets'
    # TARGET_DATASET_NAME = 'med_test' # Change as needed
    TARGET_DATASET_NAME = 'whole_dataset' # Change as needed
    BATCH_SIZE = 2
    NUM_EPOCHS = 9
    LEARNING_RATE = 5e-5
    TRAIN_TEST_SPLIT_RATIO = 0.8
    RANDOM_SEED = 42

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading BLIP processor and model from {MODEL_NAME}...")
    processor = BlipProcessor.from_pretrained(MODEL_NAME)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)

    print(f"Preparing train and test dataloaders for '{TARGET_DATASET_NAME}'...")
    train_dataloader, test_dataloader = get_multicare_dataloaders_with_split(
        dataset_directory=DATASET_DIR,
        dataset_name=TARGET_DATASET_NAME,
        processor=processor,
        batch_size=BATCH_SIZE,
        num_workers=0,
        create_new=False, # Change when running on med_test
        filters=None,
        train_ratio=TRAIN_TEST_SPLIT_RATIO,
        seed=RANDOM_SEED
    )
    print(f"Training dataloader created with {len(train_dataloader.dataset)} samples.")
    print(f"Test dataloader created with {len(test_dataloader.dataset)} samples.")

    print("\n--- Starting Training ---")
    train_blip_model(model, train_dataloader, num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE, device=device)

    print("\n--- Starting Evaluation ---")
    metrics = evaluate_blip_model(model, test_dataloader, processor, device=device)
    print("\nFinal Metrics:", metrics)

if __name__ == "__main__":
    main()

