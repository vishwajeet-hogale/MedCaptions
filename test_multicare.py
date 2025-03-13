import os
import matplotlib.pyplot as plt
from PIL import Image
from multiversity.multicare_dataset import MedicalDatasetCreator

def test_multicare_loader(dataset_name='test_sample', num_samples=1):
    """
    Test the MultiCaRe dataset loader by displaying a few sample images and their captions.
    
    Args:
        dataset_name (str): Name for the test dataset
        num_samples (int): Number of samples to display
    """
    # Initialize the MedicalDatasetCreator
    print("Initializing MedicalDatasetCreator...")
    mdc = MedicalDatasetCreator(directory='medical_datasets')
    
    # Simple filter to get a small subset of images with captions
    test_filters = [
        {'field': 'caption', 'string_list': [''], 'operator': 'any'},  # Any image with a caption
        {'field': 'label', 'string_list': ['mri'], 'operator': 'any'}  # Limit to MRI images for this test
    ]
    
    # Use display_example to see available dataset examples
    print("Displaying an example from the dataset to verify access:")
    try:
        mdc.display_example()
        print("Successfully displayed an example from the dataset")
    except Exception as e:
        print(f"Error displaying example: {e}")
    
    # List available datasets
    available_datasets = []
    try:
        # Try to get a list of datasets if such a method exists
        for method_name in ['list_datasets', 'get_datasets', 'get_dataset_names']:
            if hasattr(mdc, method_name):
                available_datasets = getattr(mdc, method_name)()
                print(f"Available datasets: {available_datasets}")
                break
        
        # If we found datasets, let's try to use an existing one
        if dataset_name in available_datasets:
            print(f"Using existing dataset: {dataset_name}")
        else:
            # Try to create a new dataset with a unique name
            new_dataset_name = f"{dataset_name}_{len(available_datasets)}"
            print(f"Creating new dataset: {new_dataset_name}")
            mdc.create_dataset(
                dataset_name=new_dataset_name,
                filter_list=test_filters,
                dataset_type='multimodal'
            )
            dataset_name = new_dataset_name
    except Exception as e:
        print(f"Error handling datasets: {e}")
        # Let's try to directly display examples without creating a dataset
        print("Displaying examples directly from the main dataset...")
        
        # Display num_samples examples
        for i in range(num_samples):
            try:
                mdc.display_example()
            except Exception as e:
                print(f"Error displaying example {i+1}: {e}")
        
        return
    
    # Based on the documentation, let's try to use the dataset we have
    print(f"\n--- Showing {num_samples} examples from the dataset ---\n")
    
    samples_shown = 0
    while samples_shown < num_samples:
        try:
            mdc.display_example()
            samples_shown += 1
        except Exception as e:
            print(f"Error displaying example: {e}")
            break
    
    if samples_shown == 0:
        print("Could not display any examples. Trying without dataset name...")
        for i in range(num_samples):
            try:
                mdc.display_example()
                samples_shown += 1
            except Exception as e:
                print(f"Error displaying example without dataset name: {e}")
                break

if __name__ == "__main__":
    # Test with a single sample
    test_multicare_loader(num_samples=1) 