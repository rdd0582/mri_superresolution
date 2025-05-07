#!/usr/bin/env python
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
import logging
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
import random

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import project modules
from scripts.extract_paired_slices import extract_slices
from scripts.infer import load_model, preprocess_image, find_best_checkpoint
from utils.losses import SSIM

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

def extract_test_slice(test_dataset_dir, hr_output_dir, lr_output_dir):
    """Extract 10 slices from test dataset, then pick one random pair."""
    logger.info(f"Extracting 10 slices from {test_dataset_dir}...")
    
    # Find a suitable nifti file in the test dataset
    nifti_file = None
    for root, dirs, files in os.walk(test_dataset_dir):
        if os.path.basename(root).lower() == "anat":
            for file in files:
                if file.endswith('.nii') or file.endswith('.nii.gz'):
                    nifti_file = os.path.join(root, file)
                    break
            if nifti_file:
                break
    
    if not nifti_file:
        logger.error("No NIfTI files found in test dataset")
        return None
    
    logger.info(f"Using NIfTI file: {nifti_file}")
    
    # Extract 10 slices
    try:
        extract_slices(
            nifti_file,
            hr_output_dir,
            lr_output_dir,
            n_slices=10,
            lower_percent=0.45,
            upper_percent=0.55,
            target_size=(256, 256),
            noise_std=5,
            kspace_crop_factor=0.5
        )
        # Find the extracted files
        hr_files = [f for f in os.listdir(hr_output_dir) if f.endswith('.png')]
        lr_files = [f for f in os.listdir(lr_output_dir) if f.endswith('.png')]
        if not hr_files or not lr_files:
            logger.error("No files were extracted")
            return None
        # Find matching pairs
        matching_pairs = [f for f in hr_files if f in lr_files]
        if not matching_pairs:
            logger.warning("No exact matching pairs found, using first files")
            return {
                'hr': os.path.join(hr_output_dir, hr_files[0]),
                'lr': os.path.join(lr_output_dir, lr_files[0])
            }
        # Pick a random pair
        chosen = random.choice(matching_pairs)
        return {
            'hr': os.path.join(hr_output_dir, chosen),
            'lr': os.path.join(lr_output_dir, chosen)
        }
    except Exception as e:
        logger.error(f"Error extracting slice from {nifti_file}: {e}")
        return None

def upscale_with_interpolation(lr_image_path, method, scale_factor=2):
    """
    Upscale image using specified interpolation method.
    
    Args:
        lr_image_path: Path to low-resolution image
        method: Interpolation method (one of 'bilinear', 'sharp_bilinear', 'bicubic')
        scale_factor: Factor to scale the image by
        
    Returns:
        Upscaled image as numpy array in range [0, 1]
    """
    # Load image
    img = cv2.imread(lr_image_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    
    # Determine interpolation method
    if method == 'bilinear':
        interp = cv2.INTER_LINEAR
    elif method == 'sharp_bilinear':
        # For sharp bilinear, we'll use bilinear with sharpening
        interp = cv2.INTER_LINEAR
    elif method == 'bicubic':
        interp = cv2.INTER_CUBIC
    else:
        raise ValueError(f"Unknown interpolation method: {method}")
        
    # Upscale image
    upscaled = cv2.resize(img, (w * scale_factor, h * scale_factor), interpolation=interp)
    
    # Apply sharpening if sharp bilinear
    if method == 'sharp_bilinear':
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        upscaled = cv2.filter2D(upscaled, -1, kernel)
        # Ensure values stay in valid range
        upscaled = np.clip(upscaled, 0, 255)
    
    # Normalize to [0, 1]
    upscaled = upscaled.astype(np.float32) / 255.0
    
    return upscaled

def upscale_with_model(model, lr_image_path, device):
    """
    Upscale image using the neural network model.
    
    Args:
        model: PyTorch model for super-resolution
        lr_image_path: Path to low-resolution image
        device: Device to run inference on
        
    Returns:
        Upscaled image as numpy array in range [0, 1]
    """
    # Load and preprocess the image
    _, input_tensor = preprocess_image(lr_image_path)
    input_tensor = input_tensor.to(device)
    
    # Run inference
    with torch.no_grad():
        output_tensor = model(input_tensor)
    
    # Convert to numpy
    output_img = output_tensor.squeeze().cpu().numpy()
    
    # Ensure values are in [0, 1]
    output_img = np.clip(output_img, 0, 1)
    
    return output_img

def calculate_metrics(hr_image, upscaled_image):
    """
    Calculate quality metrics between HR and upscaled images.
    
    Args:
        hr_image: High-resolution reference image (numpy array [0, 1])
        upscaled_image: Upscaled image to evaluate (numpy array [0, 1])
        
    Returns:
        Dictionary of metrics
    """
    # Convert to torch tensors for SSIM calculation
    # Add batch and channel dimensions
    hr_tensor = torch.from_numpy(hr_image).unsqueeze(0).unsqueeze(0)
    up_tensor = torch.from_numpy(upscaled_image).unsqueeze(0).unsqueeze(0)
    
    # Calculate SSIM
    ssim_metric = SSIM(window_size=11, sigma=1.5, val_range=1.0)
    ssim_value = ssim_metric(up_tensor, hr_tensor).item()
    
    # Calculate MSE, RMSE and MAE
    mse = ((hr_image - upscaled_image) ** 2).mean()
    rmse = np.sqrt(mse)
    mae = np.abs(hr_image - upscaled_image).mean()
    
    # Calculate PSNR (peak signal-to-noise ratio)
    # Handle potential division by zero if images are identical (MSE=0)
    if mse < 1e-10: # Use a small threshold to avoid floating point issues
        psnr = 100.0 # Assign a high finite value for perfect reconstruction
    else:
        psnr = calculate_psnr(hr_image, upscaled_image, data_range=1.0)
    
    return {
        'ssim': ssim_value,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'psnr': psnr
    }

def visualize_results(hr_image, lr_image, upscaled_images, metrics, output_path):
    """
    Create visualization of all upscaling methods with metrics.
    
    Args:
        hr_image: High-resolution reference image (numpy array [0, 1])
        lr_image: Low-resolution input image (numpy array [0, 1])
        upscaled_images: Dictionary of upscaled images by method
        metrics: Dictionary of metrics by method
        output_path: Path to save visualization
    """
    # Number of rows and columns for the plot
    n_cols = len(upscaled_images) + 2  # +2 for HR and LR
    n_rows = 2  # 1 for images, 1 for difference maps
    
    plt.figure(figsize=(n_cols * 4, n_rows * 4))
    
    # Add HR image
    plt.subplot(n_rows, n_cols, 1)
    plt.imshow(hr_image, cmap='gray', vmin=0, vmax=1)
    plt.title('HR Reference')
    plt.axis('off')
    
    # Add LR image
    plt.subplot(n_rows, n_cols, 2)
    plt.imshow(lr_image, cmap='gray', vmin=0, vmax=1)
    plt.title('LR Input')
    plt.axis('off')
    
    # Add upscaled images
    col_idx = 3
    for method, image in upscaled_images.items():
        # Add upscaled image
        plt.subplot(n_rows, n_cols, col_idx)
        plt.imshow(image, cmap='gray', vmin=0, vmax=1)
        
        # Add metrics to title
        metric_text = f'{method}\nSSIM: {metrics[method]["ssim"]:.4f}\nPSNR: {metrics[method]["psnr"]:.2f}'
        plt.title(metric_text)
        plt.axis('off')
        
        # Add difference map
        plt.subplot(n_rows, n_cols, col_idx + n_cols)
        diff = np.abs(hr_image - image)
        plt.imshow(diff, cmap='hot', vmin=0, vmax=0.5)
        plt.title(f'Difference Map\nMAE: {metrics[method]["mae"]:.4f}')
        plt.axis('off')
        
        col_idx += 1
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved visualization to {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Test and compare upscaling methods')
    parser.add_argument('--test_dataset', type=str, default='./test_dataset',
                      help='Directory containing test dataset')
    parser.add_argument('--output_dir', type=str, default='./test_results',
                      help='Directory to save test results')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                      help='Directory containing model checkpoints')
    parser.add_argument('--model_type', type=str, default='unet',
                      help='Model type to test')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    hr_dir = os.path.join(args.output_dir, 'hr')
    lr_dir = os.path.join(args.output_dir, 'lr')
    os.makedirs(hr_dir, exist_ok=True)
    os.makedirs(lr_dir, exist_ok=True)
    
    # Extract test slice
    logger.info("Extracting test slice...")
    paired_files = extract_test_slice(args.test_dataset, hr_dir, lr_dir)
    
    if not paired_files:
        logger.error("Failed to extract test slice")
        return
    
    logger.info(f"Test files: HR={paired_files['hr']}, LR={paired_files['lr']}")
    
    # Load model
    logger.info("Loading model...")
    try:
        checkpoint_path = find_best_checkpoint(args.checkpoint_dir, args.model_type)
        # Try to auto-detect base_filters from checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        base_filters = 128  # default
        if isinstance(checkpoint, dict) and 'base_filters' in checkpoint:
            base_filters = checkpoint['base_filters']
        logger.info(f"Using base_filters={base_filters}")
        model = load_model(args.model_type, checkpoint_path, device, base_filters=base_filters)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return
    
    # Load HR and LR images
    hr_image = cv2.imread(paired_files['hr'], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    lr_image = cv2.imread(paired_files['lr'], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    
    # Upscale with different methods
    logger.info("Upscaling with different methods...")
    upscaled_images = {}
    metrics = {}
    
    # 1. AI model
    logger.info("Upscaling with AI model...")
    upscaled_images['AI Model'] = upscale_with_model(model, paired_files['lr'], device)
    
    # 2. Traditional methods
    for method in ['bilinear', 'sharp_bilinear', 'bicubic']:
        logger.info(f"Upscaling with {method}...")
        upscaled_images[method.replace('_', ' ').title()] = upscale_with_interpolation(
            paired_files['lr'], method, scale_factor=2
        )
    
    # Calculate metrics
    logger.info("Calculating metrics...")
    for method, image in upscaled_images.items():
        metrics[method] = calculate_metrics(hr_image, image)
        logger.info(f"Metrics for {method}: {metrics[method]}")
    
    # Visualize results
    logger.info("Creating visualization...")
    visualize_path = os.path.join(args.output_dir, 'comparison.png')
    visualize_results(hr_image, lr_image, upscaled_images, metrics, visualize_path)
    
    # Save detailed metrics as text file
    metrics_path = os.path.join(args.output_dir, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("# Super-Resolution Comparison Metrics\n\n")
        f.write(f"Test file: {os.path.basename(paired_files['hr'])}\n\n")
        f.write("| Method | SSIM | PSNR | MSE | RMSE | MAE |\n")
        f.write("| --- | --- | --- | --- | --- | --- |\n")
        for method, method_metrics in metrics.items():
            f.write(f"| {method} | {method_metrics['ssim']:.4f} | {method_metrics['psnr']:.2f} | {method_metrics['mse']:.6f} | {method_metrics['rmse']:.4f} | {method_metrics['mae']:.4f} |\n")
    
    logger.info(f"Results saved to {args.output_dir}")
    logger.info("Testing complete!")

if __name__ == '__main__':
    main() 