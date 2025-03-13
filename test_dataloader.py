import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time

def separator(title):
    """Print a section separator with title."""
    print("\n" + "="*80)
    print(f" {title} ".center(80, "="))
    print("="*80 + "\n")

def test_all():
    """Test all components of the MultiCaRe dataset and dataloader."""
    
    # Set up matplotlib to save figures if display is not available
    plt.switch_backend('agg')
    
    # ---------------------------------------------------------
    separator("1. TESTING IMPORTS")
    # ---------------------------------------------------------
    
    try:
        print("Importing multiversity.multicare_dataset...")
        from multiversity.multicare_dataset import MedicalDatasetCreator
        print("✅ Successfully imported MedicalDatasetCreator")
    except Exception as e:
        print(f"❌ Error importing MedicalDatasetCreator: {e}")
        print("Cannot continue testing without base library.")
        return

    try:
        print("\nImporting custom dataloader components...")
        from dataloader import MultiCaReDataset, create_multicare_dataset, get_multicare_dataloader
        print("✅ Successfully imported all dataloader components")
    except Exception as e:
        print(f"❌ Error importing dataloader: {e}")
        print("Will test only base MedicalDatasetCreator functionality.")
    
    # ---------------------------------------------------------
    separator("2. TESTING MEDICAL DATASET CREATOR")
    # ---------------------------------------------------------
    
    print("Initializing MedicalDatasetCreator...")
    try:
        mdc = MedicalDatasetCreator(directory='medical_datasets')
        print("✅ Successfully initialized MedicalDatasetCreator")
        
        # Print available methods to understand the API
        print("\nAvailable methods in MedicalDatasetCreator:")
        methods = [method for method in dir(mdc) if not method.startswith('_')]
        for method in methods:
            print(f"  - {method}")
            
    except Exception as e:
        print(f"❌ Error initializing MedicalDatasetCreator: {e}")
        print("Cannot continue testing without initialized dataset.")
        return
    
    # Let's check if we can get a list of existing datasets
    print("\nLooking for existing datasets:")
    try:
        # Try common method names
        existing_datasets = []
        if hasattr(mdc, 'list_datasets'):
            existing_datasets = mdc.list_datasets()
            print(f"Found datasets using list_datasets: {existing_datasets}")
        elif hasattr(mdc, 'get_datasets'):
            existing_datasets = mdc.get_datasets()
            print(f"Found datasets using get_datasets: {existing_datasets}")
        else:
            print("Could not find method to list datasets")
    except Exception as e:
        print(f"Error listing datasets: {e}")
    
    # ---------------------------------------------------------
    separator("3. TESTING DATASET CREATION")
    # ---------------------------------------------------------
    
    test_filters = [
        {'field': 'label', 'string_list': ['mri']},
        {'field': 'caption', 'string_list': [''], 'operator': 'any'}  # Any image with caption
    ]
    
    # Generate a unique dataset name to avoid conflicts
    dataset_name = f'test_dataset_{int(time.time())}'
    
    print(f"Creating dataset: {dataset_name}...")
    try:
        mdc.create_dataset(
            dataset_name=dataset_name,
            filter_list=test_filters,
            dataset_type='multimodal'
        )
        print(f"✅ Successfully created dataset: {dataset_name}")
    except Exception as e:
        print(f"❌ Error creating dataset: {e}")
        # If this fails, try to continue with the base dataset
        dataset_name = None
    
    # ---------------------------------------------------------
    separator("4. TESTING CUSTOM DATASET CLASS")
    # ---------------------------------------------------------
    
    # Skip if custom dataloader import failed or dataset creation failed
    if ('MultiCaReDataset' not in locals() and 'MultiCaReDataset' not in globals()) or dataset_name is None:
        print("Skipping custom dataset test - MultiCaReDataset not imported or dataset creation failed.")
    else:
        print(f"Creating MultiCaReDataset instance for dataset: {dataset_name}...")
        try:
            dataset = MultiCaReDataset(
                dataset_directory='medical_datasets',
                dataset_name=dataset_name
            )
            print(f"✅ Successfully created dataset with {len(dataset)} samples")
            
            # Test getting a sample
            print("\nRetrieving one sample from dataset...")
            try:
                sample = dataset[0]
                print(f"✅ Successfully retrieved sample")
                print(f"  Image type: {type(sample['image'])}")
                print(f"  Image shape: {sample['image'].shape}")
                print(f"  Caption: {sample['caption'][:100]}..." if len(sample['caption']) > 100 else sample['caption'])
                
                # Save the image to a file
                print("\nSaving sample image to 'test_sample.png'...")
                img = sample['image']
                if isinstance(img, torch.Tensor):
                    # Convert tensor to numpy for display
                    img = img.permute(1, 2, 0).numpy()
                    # Denormalize the image
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    img = std * img + mean
                    img = np.clip(img, 0, 1)
                
                plt.figure(figsize=(8, 8))
                plt.imshow(img)
                plt.axis('off')
                plt.title(f"Sample Image: {sample['image_id']}")
                plt.savefig('test_sample.png')
                print("✅ Image saved to 'test_sample.png'")
                
            except Exception as e:
                print(f"❌ Error retrieving sample: {e}")
        
        except Exception as e:
            print(f"❌ Error creating dataset instance: {e}")
    
    # ---------------------------------------------------------
    separator("5. TESTING DATALOADER")
    # ---------------------------------------------------------
    
    # Skip if custom dataloader import failed or dataset creation failed
    if ('get_multicare_dataloader' not in locals() and 'get_multicare_dataloader' not in globals()) or dataset_name is None:
        print("Skipping dataloader test - get_multicare_dataloader not imported or dataset creation failed.")
    else:
        print(f"Creating dataloader for dataset: {dataset_name}...")
        try:
            dataloader = get_multicare_dataloader(
                dataset_name=dataset_name,
                batch_size=2,
                shuffle=True,
                num_workers=0  # Use 0 for testing to avoid multiprocessing issues
            )
            print(f"✅ Successfully created dataloader")
            
            # Test getting a batch
            print("\nRetrieving a batch from dataloader...")
            try:
                batch = next(iter(dataloader))
                print(f"✅ Successfully retrieved batch with {len(batch['image'])} images")
                print(f"  Batch image shape: {batch['image'].shape}")
                print(f"  First caption: {batch['caption'][0][:100]}..." if len(batch['caption'][0]) > 100 else batch['caption'][0])
                
                # Save the batch image to a file
                print("\nSaving batch image to 'test_batch.png'...")
                img = batch['image'][0].permute(1, 2, 0).numpy()
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = std * img + mean
                img = np.clip(img, 0, 1)
                
                plt.figure(figsize=(8, 8))
                plt.imshow(img)
                plt.axis('off')
                plt.title(f"Batch Image: {batch['image_id'][0]}")
                plt.savefig('test_batch.png')
                print("✅ Image saved to 'test_batch.png'")
                
            except Exception as e:
                print(f"❌ Error retrieving batch: {e}")
                
        except Exception as e:
            print(f"❌ Error creating dataloader: {e}")
    
    # ---------------------------------------------------------
    separator("6. CREATING DATASET WITH SPECIFIC FILTERS")
    # ---------------------------------------------------------
    
    specific_filters = [
        {'field': 'label', 'string_list': ['mri', 'head']},
        {'field': 'caption', 'string_list': ['tumor', 'mass', 'lesion'], 'operator': 'any'}
    ]
    
    # Generate a unique dataset name to avoid conflicts
    specific_dataset_name = f'brain_tumor_mri_{int(time.time())}'
    
    print(f"Creating specific dataset: {specific_dataset_name}...")
    try:
        # Using either direct MDC or helper function depending on what's available
        if 'create_multicare_dataset' in locals() or 'create_multicare_dataset' in globals():
            mdc_specific = create_multicare_dataset(
                dataset_name=specific_dataset_name,
                filters=specific_filters,
                dataset_directory='medical_datasets'
            )
            print(f"✅ Successfully created specific dataset using helper function")
        else:
            mdc.create_dataset(
                dataset_name=specific_dataset_name,
                filter_list=specific_filters,
                dataset_type='multimodal'
            )
            print(f"✅ Successfully created specific dataset directly")
        
    except Exception as e:
        print(f"❌ Error creating specific dataset: {e}")
    
    # ---------------------------------------------------------
    separator("7. TESTING COMPLETED")
    # ---------------------------------------------------------
    
    print("Testing summary:")
    print("1. Imports: Completed")
    print("2. MedicalDatasetCreator: Completed")
    print("3. Dataset Creation: Completed")
    print("4. Custom Dataset Class: Completed")
    print("5. DataLoader: Completed")
    print("6. Specific Filters: Completed")
    print("\nCheck the output above for any errors (❌) that need attention.")
    print("If images were successfully generated, check 'test_sample.png' and 'test_batch.png'.")

if __name__ == "__main__":
    test_all()