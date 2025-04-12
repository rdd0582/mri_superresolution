#!/usr/bin/env python
import argparse
import os
import sys
import json
import time
from pathlib import Path
import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import logging
import matplotlib.colors as colors

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('inference.log')
    ]
)
logger = logging.getLogger(__name__)

# Try to import AMP
try:
    from torch.amp import autocast, GradScaler
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False
    logger.warning("Automatic Mixed Precision not available. Using default precision.")

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import project modules
from utils.losses import SSIM
from utils.preprocessing import tensor_to_numpy, denormalize_from_range

def load_model(model_type, checkpoint_path, device, **kwargs):
    """Load the appropriate model with weights from checkpoint."""
    try:
        # Import the correct model based on model_type
        if model_type == "unet":
            from models.unet_model import UNetSuperRes
            model = UNetSuperRes(
                in_channels=1,
                out_channels=1,
                base_filters=kwargs.get('base_filters', 64)
            ).to(device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            model.load_state_dict(state_dict)
            logger.info(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            model.load_state_dict(checkpoint)
            logger.info(f"Loaded model weights from {checkpoint_path}")
        
        model.eval()
        return model
    
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def find_best_checkpoint(checkpoint_dir, model_type):
    """Find the best checkpoint for the specified model type."""
    # First try to find best_model_[model_type].pth
    best_path = os.path.join(checkpoint_dir, f"best_model_{model_type}.pth")
    if os.path.exists(best_path):
        logger.info(f"Using best model checkpoint: {best_path}")
        return best_path
    
    # Then try final_model_[model_type].pth
    final_path = os.path.join(checkpoint_dir, f"final_model_{model_type}.pth")
    if os.path.exists(final_path):
        logger.info(f"Using final model checkpoint: {final_path}")
        return final_path
    
    # Look for any .pth file containing the model_type name
    for file in os.listdir(checkpoint_dir):
        if file.endswith('.pth') and model_type in file:
            path = os.path.join(checkpoint_dir, file)
            logger.info(f"Using found checkpoint: {path}")
            return path
    
    raise FileNotFoundError(f"No checkpoint found for {model_type} model in {checkpoint_dir}")

def preprocess_image(image_path):
    """Load and preprocess an image for inference."""
    try:
        # Load image
        image = Image.open(image_path).convert('L')
        
        # Convert to numpy array for processing
        image_np = np.array(image).astype(np.float32)
        
        # Apply the same preprocessing as during training data generation:
        # 1. Apply percentile-based windowing/clipping
        min_percentile = 0.5  # Same as used in extract_paired_slices.py -> preprocess_slice
        max_percentile = 99.5
        min_val = np.percentile(image_np, min_percentile)
        max_val = np.percentile(image_np, max_percentile)
        image_np = np.clip(image_np, min_val, max_val)
        
        # 2. Normalize to [0, 1] range
        if max_val > min_val:
            image_np = (image_np - min_val) / (max_val - min_val)
        
        # Convert to tensor
        tensor = torch.from_numpy(image_np).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        
        return image, tensor
    
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        raise

def postprocess_tensor(tensor):
    """Convert tensor to PIL Image."""
    # Remove batch dimension and move to CPU
    tensor = tensor.squeeze(0).cpu()
    
    # Convert to numpy and ensure in [0, 1] range
    np_img = tensor_to_numpy(tensor)
    
    # Already in [0, 1] range, no need to denormalize
    
    # Convert to uint8 for PIL
    np_img = (np_img * 255).astype(np.uint8)
    
    # Create PIL image
    return Image.fromarray(np_img.squeeze())

def calculate_metrics(output_tensor, target_tensor):
    """Calculate image quality metrics."""
    metrics = {}
    
    # Ensure tensors have batch dimension
    if output_tensor.dim() == 3:
        output_tensor = output_tensor.unsqueeze(0)
    if target_tensor.dim() == 3:
        target_tensor = target_tensor.unsqueeze(0)
    
    try:
        # Calculate SSIM
        ssim_metric = SSIM(window_size=11, sigma=1.5, val_range=1.0)
        metrics['ssim'] = ssim_metric(output_tensor, target_tensor).item()
        
        # Calculate MSE and MAE
        mse = torch.nn.functional.mse_loss(output_tensor, target_tensor).item()
        metrics['rmse'] = np.sqrt(mse)
        metrics['mae'] = torch.nn.functional.l1_loss(output_tensor, target_tensor).item()
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
    
    return metrics

def visualize_results(input_image, output_image, target_image=None, metrics=None, show_diff=False, save_path=None):
    """Visualize the results with optional difference map."""
    has_target = target_image is not None
    n_cols = 2 + int(has_target) + int(has_target and show_diff)
    
    plt.figure(figsize=(n_cols * 4, 5))
    
    # Plot input image
    plt.subplot(1, n_cols, 1)
    plt.imshow(input_image, cmap='gray')
    plt.title('Input Low-Resolution')
    plt.axis('off')
    
    # Plot output image
    plt.subplot(1, n_cols, 2)
    plt.imshow(output_image, cmap='gray')
    plt.title('Super-Resolution Output')
    plt.axis('off')
    
    # Plot target image if available
    if has_target:
        plt.subplot(1, n_cols, 3)
        plt.imshow(target_image, cmap='gray')
        plt.title('Ground Truth')
        plt.axis('off')
    
    # Plot difference map if requested
    if has_target and show_diff:
        # Convert images to numpy arrays
        output_np = np.array(output_image).astype(np.float32) / 255.0
        target_np = np.array(target_image).astype(np.float32) / 255.0
        
        # Calculate absolute difference
        diff = np.abs(output_np - target_np)
        
        # Plot difference map with heatmap
        plt.subplot(1, n_cols, 4)
        im = plt.imshow(diff, cmap='hot', vmin=0, vmax=0.5)
        plt.title('Absolute Difference')
        plt.axis('off')
        plt.colorbar(im, fraction=0.046, pad=0.04)
    
    # Add metrics as text if available
    if metrics:
        metrics_text = "\n".join([f"{k.upper()}: {v:.4f}" for k, v in metrics.items()])
        plt.figtext(0.5, 0.01, metrics_text, ha='center', fontsize=12, 
                   bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved visualization to {save_path}")
    
    plt.show()

def process_single_image(model, input_path, output_path, target_path=None, device="cpu", 
                        show_comparison=False, show_diff=False, use_amp=False):
    """
    Process a single image for super-resolution.
    """
    # Load and preprocess input image
    image, tensor = preprocess_image(input_path)
    tensor = tensor.to(device)
    
    # Process target image if provided
    target_tensor = None
    metrics = None  # Initialize metrics to None
    if target_path and os.path.exists(target_path):
        _, target_tensor = preprocess_image(target_path)
        target_tensor = target_tensor.to(device)
    
    # Model inference
    model.eval()
    with torch.no_grad():
        if use_amp and AMP_AVAILABLE and device.type == 'cuda':
            with autocast('cuda'):
                output_tensor = model(tensor)
        else:
            output_tensor = model(tensor)
    
    # Calculate metrics if target is available
    if target_tensor is not None:
        metrics = calculate_metrics(output_tensor, target_tensor)
        for metric_name, metric_value in metrics.items():
            logger.info(f"{metric_name.upper()}: {metric_value:.4f}")
    
    # Convert output tensor to image and save
    output_image = postprocess_tensor(output_tensor)
    output_image.save(output_path)
    logger.info(f"Enhanced image saved to {output_path}")
    
    # Display comparison if requested
    if show_comparison:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3 if target_tensor is not None else 2, 1)
        plt.title("Input (Low Quality)")
        plt.imshow(np.array(image), cmap='gray')
        plt.axis('off')
        
        plt.subplot(1, 3 if target_tensor is not None else 2, 2)
        plt.title("Output (Enhanced)")
        plt.imshow(np.array(output_image), cmap='gray')
        plt.axis('off')
        
        if target_tensor is not None:
            target_image = postprocess_tensor(target_tensor)
            plt.subplot(1, 3, 3)
            plt.title("Target (High Quality)")
            plt.imshow(np.array(target_image), cmap='gray')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    # Show difference map if requested
    if show_diff and target_tensor is not None:
        target_image = postprocess_tensor(target_tensor)
        
        # Calculate difference
        diff = np.abs(np.array(output_image).astype(np.float32) - 
                     np.array(target_image).astype(np.float32))
        
        # Normalize and colorize difference
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.title("Enhanced Output")
        plt.imshow(np.array(output_image), cmap='gray')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.title("Target")
        plt.imshow(np.array(target_image), cmap='gray')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.title("Difference Map")
        plt.imshow(diff, cmap='hot')
        plt.colorbar(label='Absolute Difference')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return output_image, metrics

def main(args):
    """Main inference function."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    logger.info(f"Using device: {device}")
    
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        logger.info(f"Memory reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    
    # Check if AMP can be used
    use_amp = args.use_amp and AMP_AVAILABLE and device.type == 'cuda'
    if args.use_amp and not use_amp:
        logger.warning("AMP requested but not available. Falling back to full precision.")
    if use_amp:
        logger.info("Using Automatic Mixed Precision (AMP) for inference.")
    
    try:
        # Find checkpoint or use the one specified
        if args.checkpoint_path and os.path.exists(args.checkpoint_path):
            checkpoint_path = args.checkpoint_path
            logger.info(f"Using specified checkpoint: {checkpoint_path}")
        else:
            checkpoint_path = find_best_checkpoint(args.checkpoint_dir, args.model_type)
            logger.info(f"Automatically selected checkpoint: {checkpoint_path}")
        
        # Load model
        model = load_model(
            args.model_type, 
            checkpoint_path, 
            device,
            base_filters=args.base_filters,
        )
        
        # Process a single image
        process_single_image(
            model=model,
            input_path=args.input,
            output_path=args.output,
            target_path=args.target,
            device=device,
            show_comparison=args.show_comparison,
            show_diff=args.show_diff,
            use_amp=use_amp
        )
            
        logger.info("Inference completed successfully!")
        return 0
    
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        return 1

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="MRI quality enhancement inference")
    
    # Paths
    parser.add_argument('--input', type=str, required=True,
                      help='Path to input image')
    parser.add_argument('--output', type=str, required=True,
                      help='Path to output image')
    parser.add_argument('--target', type=str, default=None,
                      help='Path to target image (for comparison)')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                      help='Directory containing model checkpoints')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                      help='Specific checkpoint file path to use (overrides automatic checkpoint finding)')
    
    # Model selection parameter
    parser.add_argument('--model_type', type=str, choices=['unet'], default='unet',
                      help='Model architecture to use (only unet is supported)')
    
    # Model-specific parameters
    parser.add_argument('--base_filters', type=int, default=64,
                      help='Number of base filters in the UNet model')
    
    # Inference options
    parser.add_argument('--show_comparison', action='store_true', 
                        help="Show comparison visualization")
    parser.add_argument('--show_diff', action='store_true', 
                        help="Show difference map in visualization")
    parser.add_argument('--cpu', action='store_true', 
                        help="Force using CPU even if CUDA is available")
    parser.add_argument('--use_amp', action='store_true',
                        help="Use Automatic Mixed Precision for faster inference on RTX GPUs")
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args)
