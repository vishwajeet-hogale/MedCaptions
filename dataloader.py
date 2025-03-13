import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from multiversity.multicare_dataset import MedicalDatasetCreator
import json

class MultiCaReDataset(Dataset):
    """Dataset class for loading MultiCaRe medical images and captions for image captioning."""
    
    def __init__(self, dataset_directory='medical_datasets', dataset_name='brain_tumor_mri', 
                 transform=None, create_dataset=False, filters=None):
        """
        Initialize the MultiCaRe dataset for image captioning.
        
        Args:
            dataset_directory (str): Directory where the MultiCaRe data is stored
            dataset_name (str): Name of the specific dataset
            transform (callable, optional): Optional transform to apply to images
            create_dataset (bool): Whether to create a new dataset
            filters (list): Filters to use when creating a new dataset
        """
        self.dataset_directory = dataset_directory
        self.dataset_name = dataset_name
        self.transform = transform
        
        # Create dataset if requested
        if create_dataset:
            if not filters:
                # Default filters to get MRI images with captions
                filters = [
                    {'field': 'label', 'string_list': ['mri', 'head']},
                    {'field': 'caption', 'string_list': [''], 'operator': 'any'}  # Any image with caption
                ]
            create_multicare_dataset(dataset_name, filters, dataset_directory)
        
        # Set up default image transforms if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        # Load the dataset metadata
        all_data = self._load_metadata()
        
        # Filter out entries without captions
        captioned_data = [item for item in all_data if item.get('caption', '').strip()]
        print(f"Filtered to {len(captioned_data)} samples with captions")
        
        # Filter out entries with missing images
        self.data = self._filter_valid_images(captioned_data)
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
        
        # Print diagnostic info for the first few missing items
        global missing_image_counter
        if not hasattr(self, 'missing_image_counter'):
            self.missing_image_counter = 0
        
        if self.missing_image_counter < 5:  # Only print for the first 5 missing images
            print(f"Missing image sample data: {sample_data}")
            self.missing_image_counter += 1
        
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
            
        # Get data for this sample
        sample_data = self.data[idx]
        
        # Use the pre-resolved image path (faster than re-resolving)
        image_path = sample_data['_resolved_image_path']
            
        # Load the image (we know it exists at this point)
        try:
            image = Image.open(image_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
                
            # Extract the caption
            caption = sample_data.get('caption', '')
            
            # For image captioning, we only need the image and caption, plus an ID for reference
            result = {
                'image': image,
                'caption': caption,
                'image_id': sample_data.get('file_id', sample_data.get('main_image', f"unknown_{idx}"))
            }
            
            return result
            
        except Exception as e:
            # This should rarely happen since we pre-validated the images
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
    if os.path.exists(dataset_dir):
        print(f"Dataset directory '{dataset_name}' already exists, using existing version.")
        mdc = MedicalDatasetCreator(directory=dataset_directory)
        return mdc
    
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
        raise ValueError("Filters must be provided to create a new dataset")
    
    return mdc

def get_multicare_dataloader(dataset_name, batch_size=16, shuffle=True, num_workers=4, 
                            dataset_directory='medical_datasets', create_new=False, filters=None):
    """
    Create a DataLoader for a MultiCaRe dataset optimized for image captioning.
    
    Args:
        dataset_name (str): Name of the dataset to use
        batch_size (int): Batch size for training
        shuffle (bool): Whether to shuffle the dataset
        num_workers (int): Number of worker threads for loading data
        dataset_directory (str): Directory where MultiCaRe data is stored
        create_new (bool): Whether to create a new dataset
        filters (list): Filters to use when creating a new dataset
        
    Returns:
        DataLoader: PyTorch DataLoader object for the MultiCaRe dataset
    """
    # Check if the dataset already exists
    dataset_dir = os.path.join(dataset_directory, dataset_name)
    
    # Create the dataset only if create_new is True and the dataset doesn't exist
    create_dataset = create_new and not os.path.exists(dataset_dir)
    
    dataset = MultiCaReDataset(
        dataset_directory=dataset_directory,
        dataset_name=dataset_name,
        create_dataset=create_dataset,
        filters=filters
    )
    
    # Check if dataset is empty
    if len(dataset) == 0:
        raise ValueError(f"Dataset '{dataset_name}' has 0 valid samples after filtering. Check image paths and make sure images are accessible.")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


# Example usage for image captioning:
if __name__ == "__main__":
    # Define filters for medical images with descriptive captions
    caption_filters = [
        {'field': 'label', 'string_list': ['mri', 'head']},
        # Get images with meaningful captions (avoid empty captions)
        {'field': 'caption', 'string_list': ['showing', 'demonstrates', 'reveals'], 'operator': 'any'}
    ]
    
    # Print the dataset directory structure to help debugging
    print("\nExamining directory structure:")
    dataset_dir = 'medical_datasets/med_test'
    if os.path.exists(dataset_dir):
        print(f"Contents of {dataset_dir}:")
        for item in os.listdir(dataset_dir):
            print(f"  - {item}")
            if item == 'images' and os.path.isdir(os.path.join(dataset_dir, 'images')):
                image_dir = os.path.join(dataset_dir, 'images')
                print(f"    Contents of {image_dir} (first 5 items):")
                for img in list(os.listdir(image_dir))[:5]:
                    print(f"      - {img}")
    
    try:
        # Create and get a dataloader
        loader = get_multicare_dataloader(
            dataset_name='med_test',
            batch_size=8,
            create_new=True,
            filters=caption_filters
        )
        
        # Print sample batch
        for batch in loader:
            print(f"Batch size: {len(batch['image'])}")
            print(f"Image shape: {batch['image'].shape}")
            print(f"Sample caption: {batch['caption'][0]}")
            break
    except Exception as e:
        print(f"Error creating dataloader: {e}")
        print("\nIf you're having issues with image paths, try these suggestions:")
        print("1. Check if images directory exists in your dataset folder")
        print("2. Verify that images are in the expected location")
        print("3. Try using the MultiCaRe examples to create a dataset first")
