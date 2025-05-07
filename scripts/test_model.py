#!/usr/bin/env python
import os
import sys
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from PIL import Image
import logging
from tqdm import tqdm
import nibabel as nib

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import from other scripts
from scripts.extract_paired_slices import extract_slices
from scripts.infer import load_model, process_single_image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_model.log')
    ]
)
logger = logging.getLogger(__name__)

def extract_test_slices(test_dataset_dir, hr_output_dir, lr_output_dir, n_slices=10):
    """Extract slices from test dataset"""
    logger.info(f"Extracting {n_slices} slices from {test_dataset_dir}...")
    
    # Create output directories
    os.makedirs(hr_output_dir, exist_ok=True)
    os.makedirs(lr_output_dir, exist_ok=True)
    
    # Find all .nii or .nii.gz files in the test_dataset_dir, but only in "anat" folders
    nifti_files = []
    for root, _, files in os.walk(test_dataset_dir):
        # Only process directories named "anat"
        if os.path.basename(root).lower() != "anat":
            continue
            
        for file in files:
            if file.endswith('.nii') or file.endswith('.nii.gz'):
                nifti_files.append(os.path.join(root, file))
    
    if not nifti_files:
        logger.error(f"No NIfTI files found in 'anat' folders within {test_dataset_dir}")
        return None
    
    logger.info(f"Found {len(nifti_files)} NIfTI files in 'anat' folders.")
    
    # First, calculate the average resolution of the slices
    total_width = 0
    total_height = 0
    total_slices = 0
    
    # Analyze each NIfTI file to calculate average resolution
    logger.info("Calculating average slice resolution...")
    for nifti_file in nifti_files:
        try:
            img = nib.load(nifti_file)
            data = img.get_fdata()
            
            if data.ndim >= 3:  # Ensure it's a 3D or 4D image
                # Get width and height (first two dimensions)
                height, width = data.shape[:2]
                # Count slices (third dimension)
                slices_count = data.shape[2]
                
                total_width += width * slices_count
                total_height += height * slices_count
                total_slices += slices_count
                
                logger.info(f"File {os.path.basename(nifti_file)}: {width}x{height}, {slices_count} slices")
        except Exception as e:
            logger.error(f"Error analyzing resolution for {nifti_file}: {e}")
    
    if total_slices == 0:
        logger.error("No valid slices found in NIfTI files.")
        return None
    
    # Calculate average resolution
    avg_width = int(total_width / total_slices)
    avg_height = int(total_height / total_slices)
    logger.info(f"Average slice resolution: {avg_width}x{avg_height}")
    
    # For HR, use the average resolution but make it square and divisible by 8
    # 1. Find the larger dimension to make it square
    hr_size = max(avg_width, avg_height)
    
    # 2. Make it divisible by 8 (round up)
    if hr_size % 8 != 0:
        hr_size = ((hr_size // 8) + 1) * 8
    
    # 3. Set the target size to be square
    hr_target_size = (hr_size, hr_size)
    logger.info(f"Setting HR target size to square and divisible by 8: {hr_size}x{hr_size}")
    
    # For LR, set to half the HR size
    lr_size = hr_size // 2
    lr_target_size = (lr_size, lr_size)
    logger.info(f"Setting LR target size to half: {lr_size}x{lr_size}")
    
    # We need to modify the extract_slices function to use zero-padding
    # Since we can't modify it directly, we'll handle the resizing and padding after extraction
    
    # First extract the slices using the original function with approximate sizes
    for nifti_file in tqdm(nifti_files):
        try:
            extract_slices(
                nifti_file, 
                hr_output_dir, 
                lr_output_dir,
                n_slices=n_slices // len(nifti_files) + 1,
                lower_percent=0.2, 
                upper_percent=0.8,
                target_size=(avg_width, avg_height)  # Use average size initially
            )
        except Exception as e:
            logger.error(f"Error extracting slices from {nifti_file}: {e}")
    
    # Now resize and pad all extracted images to make them square and divisible by 8
    logger.info("Post-processing extracted slices to make square and divisible by 8...")
    
    # Process HR images
    hr_files = [f for f in os.listdir(hr_output_dir) if f.endswith('.png')]
    for hr_file in tqdm(hr_files, desc="Processing HR slices"):
        file_path = os.path.join(hr_output_dir, hr_file)
        img = Image.open(file_path)
        
        # Create a new square black image (zero-padded) with the target size
        new_img = Image.new('L', hr_target_size, color=0)
        
        # Calculate position to paste the original image (centered)
        paste_x = (hr_size - img.width) // 2
        paste_y = (hr_size - img.height) // 2
        
        # Paste the original image onto the padded canvas
        new_img.paste(img, (paste_x, paste_y))
        
        # Save the padded image
        new_img.save(file_path)
    
    # Process LR images
    lr_files = [f for f in os.listdir(lr_output_dir) if f.endswith('.png')]
    for lr_file in tqdm(lr_files, desc="Processing LR slices"):
        file_path = os.path.join(lr_output_dir, lr_file)
        img = Image.open(file_path)
        
        # Create a new square black image with the target size
        new_img = Image.new('L', lr_target_size, color=0)
        
        # Calculate position to paste the original image (centered)
        paste_x = (lr_size - img.width) // 2
        paste_y = (lr_size - img.height) // 2
        
        # Paste the original image onto the padded canvas
        new_img.paste(img, (paste_x, paste_y))
        
        # Save the padded image
        new_img.save(file_path)
    
    # Return list of paired files
    hr_files = [f for f in os.listdir(hr_output_dir) if f.endswith('.png')]
    lr_files = [f for f in os.listdir(lr_output_dir) if f.endswith('.png')]
    
    # Find matching pairs by basename
    paired_files = []
    for hr_file in hr_files:
        if hr_file in lr_files:
            paired_files.append((
                os.path.join(lr_output_dir, hr_file),  # LR input
                os.path.join(hr_output_dir, hr_file)   # HR target
            ))
    
    # Select at most n_slices pairs
    if len(paired_files) > n_slices:
        paired_files = random.sample(paired_files, n_slices)
    
    logger.info(f"Extracted {len(paired_files)} paired slices for testing")
    return paired_files

def test_model(model, paired_files, output_dir, device, use_amp=False):
    """Run inference on extracted slices and collect results"""
    os.makedirs(output_dir, exist_ok=True)
    results = []
    
    for i, (lr_file, hr_file) in enumerate(tqdm(paired_files)):
        logger.info(f"Processing slice {i+1}/{len(paired_files)}: {os.path.basename(lr_file)}")
        
        # Generate output path
        output_file = os.path.join(output_dir, f"enhanced_{os.path.basename(lr_file)}")
        
        # Process image
        try:
            _, metrics = process_single_image(
                model=model,
                input_path=lr_file,
                output_path=output_file,
                target_path=hr_file,
                device=device,
                show_comparison=False,
                show_diff=False,
                use_amp=use_amp
            )
            
            results.append({
                'input': lr_file,
                'target': hr_file,
                'output': output_file,
                'metrics': metrics
            })
            
        except Exception as e:
            logger.error(f"Error processing {lr_file}: {e}")
    
    return results

def create_summary_visualization(results, output_path):
    """Create a grid visualization of all results"""
    n_results = len(results)
    if n_results == 0:
        logger.error("No results to visualize")
        return
    
    # Determine grid size
    cols = min(4, n_results)
    rows = (n_results + cols - 1) // cols * 3  # 3 rows per result (input, output, target)
    
    plt.figure(figsize=(cols * 5, rows * 5))  # Increased figure size for better visibility
    
    # Calculate average metrics
    avg_metrics = {}
    for result in results:
        if result['metrics']:
            for k, v in result['metrics'].items():
                avg_metrics[k] = avg_metrics.get(k, 0) + v
    
    for k in avg_metrics:
        avg_metrics[k] /= len(results)
    
    # Add title with average metrics
    title = "Model Evaluation Results\n"
    if avg_metrics:
        title += " | ".join([f"{k.upper()}: {v:.4f}" for k, v in avg_metrics.items()])
    plt.suptitle(title, fontsize=16)
    
    # Plot each result
    for i, result in enumerate(results):
        row_idx = (i // cols) * 3
        col_idx = i % cols
        
        # Load images
        input_img = np.array(Image.open(result['input']))
        output_img = np.array(Image.open(result['output']))
        target_img = np.array(Image.open(result['target']))
        
        # Plot input image - disable interpolation for raw pixel display
        plt.subplot(rows, cols, row_idx * cols + col_idx + 1)
        plt.imshow(input_img, cmap='gray', interpolation='none')
        plt.title(f"Input {i+1}")
        plt.axis('off')
        
        # Plot output image - disable interpolation for raw pixel display
        plt.subplot(rows, cols, (row_idx + 1) * cols + col_idx + 1)
        plt.imshow(output_img, cmap='gray', interpolation='none')
        if result['metrics']:
            metrics_text = "\n".join([f"{k.upper()}: {v:.4f}" for k, v in result['metrics'].items()])
            plt.title(f"Output {i+1}\n{metrics_text}", fontsize=8)
        else:
            plt.title(f"Output {i+1}")
        plt.axis('off')
        
        # Plot target image - disable interpolation for raw pixel display
        plt.subplot(rows, cols, (row_idx + 2) * cols + col_idx + 1)
        plt.imshow(target_img, cmap='gray', interpolation='none')
        plt.title(f"Target {i+1}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Make room for the title
    plt.savefig(output_path, dpi=300, bbox_inches='tight')  # Increased DPI for better detail
    logger.info(f"Saved visualization to {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Test MRI super-resolution model on new dataset")
    
    parser.add_argument('--test_dataset', type=str, default='./test_dataset',
                      help='Path to test dataset directory containing NIfTI files')
    parser.add_argument('--output_dir', type=str, default='./test_results',
                      help='Directory to save results')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                      help='Directory containing model checkpoints')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                      help='Specific checkpoint file to use')
    parser.add_argument('--model_type', type=str, choices=['unet'], default='unet',
                      help='Model architecture to use')
    parser.add_argument('--base_filters', type=int, default=32,
                      help='Number of base filters in the UNet model')
    parser.add_argument('--n_slices', type=int, default=10,
                      help='Number of slices to extract for testing')
    parser.add_argument('--cpu', action='store_true',
                      help='Force using CPU even if CUDA is available')
    parser.add_argument('--use_amp', action='store_true',
                      help='Use Automatic Mixed Precision for faster inference')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    logger.info(f"Using device: {device}")
    
    try:
        # Create output directories
        os.makedirs(args.output_dir, exist_ok=True)
        hr_output_dir = os.path.join(args.output_dir, 'hr_slices')
        lr_output_dir = os.path.join(args.output_dir, 'lr_slices')
        enhanced_dir = os.path.join(args.output_dir, 'enhanced')
        
        # Extract test slices
        paired_files = extract_test_slices(
            args.test_dataset,
            hr_output_dir,
            lr_output_dir,
            n_slices=args.n_slices
        )
        
        if not paired_files:
            logger.error("No paired slices extracted. Exiting.")
            return 1
        
        # Load model
        if args.checkpoint_path:
            checkpoint_path = args.checkpoint_path
        else:
            # Find best checkpoint
            checkpoint_files = [f for f in os.listdir(args.checkpoint_dir) 
                             if f.endswith('.pth') and args.model_type in f]
            if not checkpoint_files:
                logger.error(f"No checkpoint found in {args.checkpoint_dir}")
                return 1
            
            # Prefer best_model > final_model > latest
            if f"best_model_{args.model_type}.pth" in checkpoint_files:
                checkpoint_path = os.path.join(args.checkpoint_dir, f"best_model_{args.model_type}.pth")
            elif f"final_model_{args.model_type}.pth" in checkpoint_files:
                checkpoint_path = os.path.join(args.checkpoint_dir, f"final_model_{args.model_type}.pth")
            else:
                checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_files[0])
        
        logger.info(f"Loading model from {checkpoint_path}")
        model = load_model(
            args.model_type,
            checkpoint_path,
            device,
            base_filters=args.base_filters
        )
        
        # Run inference on paired slices
        results = test_model(
            model,
            paired_files,
            enhanced_dir,
            device,
            use_amp=args.use_amp
        )
        
        # Create visualizations
        if results:
            visualization_path = os.path.join(args.output_dir, 'results_summary.png')
            create_summary_visualization(results, visualization_path)
            
            # Print summary statistics
            logger.info("=== Testing Results Summary ===")
            avg_metrics = {}
            for result in results:
                if result['metrics']:
                    for k, v in result['metrics'].items():
                        avg_metrics[k] = avg_metrics.get(k, 0) + v
            
            for k in avg_metrics:
                avg_metrics[k] /= len(results)
                logger.info(f"Average {k.upper()}: {avg_metrics[k]:.4f}")
        
        logger.info("Testing completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main()) 