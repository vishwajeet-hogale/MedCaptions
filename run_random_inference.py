#!/usr/bin/env python
import os
import glob
import random
import subprocess
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Run inference on a random medical image')
    parser.add_argument('--dataset_path', type=str, 
                        default='medical_datasets/medCapAll2/images',
                        help='Path to the dataset directory containing images')
    parser.add_argument('--output_dir', type=str, 
                        default='inference_results',
                        help='Directory to save inference results')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Get the absolute path to the dataset
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, args.dataset_path)
    
    # Find all image files
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
        pattern = os.path.join(dataset_path, '**', ext)
        image_files.extend(glob.glob(pattern, recursive=True))
    
    if not image_files:
        print(f"No image files found in {dataset_path}")
        return
    
    # Select a random image
    random_image = random.choice(image_files)
    print(f"Selected random image: {random_image}")
    
    # Run inference
    inference_cmd = ["python", "inference.py", "--image_path", random_image, "--output_dir", args.output_dir]
    print(f"Running: {' '.join(inference_cmd)}")
    subprocess.run(inference_cmd)

if __name__ == "__main__":
    main() 