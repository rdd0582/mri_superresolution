#!/usr/bin/env python
import os
import subprocess
import argparse
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def run_training_with_ssim_weight(args, ssim_weight, output_dir):
    """Run the training script with a specific SSIM weight"""
    # Create a unique checkpoint directory for this run
    weight_dir = os.path.join(output_dir, f"ssim_weight_{ssim_weight}")
    os.makedirs(weight_dir, exist_ok=True)
    
    # Construct the command to run the training script
    cmd = [
        "python", "scripts/train.py",
        "--full_res_dir", args.full_res_dir,
        "--low_res_dir", args.low_res_dir,
        "--model_type", args.model_type,
        "--batch_size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--ssim_weight", str(ssim_weight),
        "--checkpoint_dir", weight_dir,
        "--log_dir", os.path.join(weight_dir, "logs"),
    ]
    
    # Add optional arguments
    if args.augmentation:
        cmd.append("--augmentation")
    if args.use_amp:
        cmd.append("--use_amp")
    if args.cpu:
        cmd.append("--cpu")
        
    # Run the command
    print(f"Starting training with SSIM weight: {ssim_weight}")
    subprocess.run(cmd, check=True)
    
    return weight_dir

def create_ssim_weight_collage(weight_dirs, output_path, epoch=-1):
    """
    Create a collage comparing results from different SSIM weights
    
    Args:
        weight_dirs: Dictionary mapping SSIM weights to their output directories
        output_path: Path to save the collage
        epoch: Specific epoch to compare, or -1 for the last epoch
    """
    # Sort weights for consistent display
    ssim_weights = sorted(weight_dirs.keys())
    num_weights = len(ssim_weights)
    
    # Determine rows and columns for the figure
    fig = plt.figure(figsize=(15, 5 * num_weights))
    
    for i, weight in enumerate(ssim_weights):
        # Find the sample images for this weight
        sample_dir = os.path.join(weight_dirs[weight], 'samples')
        if not os.path.exists(sample_dir):
            print(f"Warning: No samples found for SSIM weight {weight}")
            continue
            
        # Get the comparison images - either specific epoch or the last one
        if epoch >= 0:
            image_path = os.path.join(sample_dir, f'comparison_epoch_{epoch}.png')
        else:
            # Find the latest epoch
            image_files = glob.glob(os.path.join(sample_dir, 'comparison_epoch_*.png'))
            if not image_files:
                print(f"Warning: No comparison images found for SSIM weight {weight}")
                continue
            image_path = max(image_files, key=os.path.getctime)
        
        # Load the image
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} not found")
            continue
            
        # Add the image to the plot
        ax = fig.add_subplot(num_weights, 1, i + 1)
        img = plt.imread(image_path)
        ax.imshow(img)
        ax.set_title(f"SSIM Weight: {weight}")
        ax.axis('off')
            
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Collage saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Test various SSIM weights for MRI Super-resolution")
    
    # Data paths (required for training)
    parser.add_argument('--full_res_dir', type=str, required=True,
                      help='Directory containing high-quality MRI slices')
    parser.add_argument('--low_res_dir', type=str, required=True,
                      help='Directory containing low-quality MRI slices')
    
    # SSIM weights to test
    parser.add_argument('--ssim_weights', type=float, nargs='+', default=[0.0, 0.3, 0.5, 0.7, 1.0],
                      help='SSIM weights to test (default: 0.0, 0.3, 0.5, 0.7, 1.0)')
    
    # Training parameters 
    parser.add_argument('--model_type', type=str, choices=['simple', 'edsr', 'unet'], default='unet',
                      help='Model architecture to use')
    parser.add_argument('--batch_size', type=int, default=8,
                      help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20,
                      help='Number of epochs to train (lower value for quicker testing)')
    
    # Options
    parser.add_argument('--augmentation', action='store_true',
                      help='Enable data augmentation')
    parser.add_argument('--use_amp', action='store_true',
                        help='Use Automatic Mixed Precision training')
    parser.add_argument('--cpu', action='store_true',
                      help='Force using CPU even if CUDA is available')
    
    # Output directory
    parser.add_argument('--output_dir', type=str, default='./ssim_weight_comparison',
                      help='Directory to save all outputs')
    
    args = parser.parse_args()
    
    # Create the main output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run training for each SSIM weight
    weight_dirs = {}
    for weight in args.ssim_weights:
        weight_dir = run_training_with_ssim_weight(args, weight, output_dir)
        weight_dirs[weight] = weight_dir
    
    # Create the collage of final results
    collage_path = os.path.join(output_dir, "ssim_weight_comparison.png")
    create_ssim_weight_collage(weight_dirs, collage_path)
    
    print(f"\nAll trainings completed. Results saved to {output_dir}")

if __name__ == "__main__":
    main() 