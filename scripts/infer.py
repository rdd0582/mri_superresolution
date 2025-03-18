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

# Check for torch.compile availability (torch 2.0+)
try:
    from torch import _dynamo
    from torch import compile
    COMPILE_AVAILABLE = True
except ImportError:
    COMPILE_AVAILABLE = False
    logger.warning("torch.compile not available. Using standard model execution.")

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
        elif model_type == "edsr":
            from models.edsr_model import EDSRSuperRes
            model = EDSRSuperRes(
                in_channels=1,
                out_channels=1,
                scale=kwargs.get('scale', 1),
                num_features=kwargs.get('num_features', 64),
                num_res_blocks=kwargs.get('num_res_blocks', 16)
            ).to(device)
        elif model_type == "simple":
            from models.cnn_model import CNNSuperRes
            model = CNNSuperRes(
                in_channels=1,
                out_channels=1,
                num_features=kwargs.get('num_features', 64),
                num_blocks=kwargs.get('num_blocks', 8)
            ).to(device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            # Check if the state dict has keys with _orig_mod prefix (from torch.compile)
            if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
                # Strip the _orig_mod prefix from all keys
                state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            logger.info(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            # Check if the state dict has keys with _orig_mod prefix (from torch.compile)
            if any(k.startswith('_orig_mod.') for k in checkpoint.keys()):
                # Strip the _orig_mod prefix from all keys
                checkpoint = {k.replace('_orig_mod.', ''): v for k, v in checkpoint.items()}
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
        
        # Convert to tensor and normalize to [-1, 1]
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        tensor = transform(image).unsqueeze(0)  # Add batch dimension
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
    
    # Denormalize from [-1, 1] to [0, 1] using the consistent helper function
    np_img = denormalize_from_range(np_img, low=-1, high=1)
    
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
        ssim_metric = SSIM(window_size=11, sigma=1.5, val_range=2.0)
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

def process_batch(model, input_dir, output_dir, device, target_dir=None, save_visualizations=False, use_amp=False):
    """Process a batch of images."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect image paths
    input_files = sorted([f for f in os.listdir(input_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
    
    # Setup metrics collection
    metrics_sum = {}
    metrics_count = 0
    
    # Process each image
    start_time = time.time()
    for i, img_file in enumerate(input_files):
        input_path = os.path.join(input_dir, img_file)
        output_path = os.path.join(output_dir, img_file)
        
        target_path = None
        if target_dir and os.path.exists(target_dir):
            potential_target = os.path.join(target_dir, img_file)
            if os.path.exists(potential_target):
                target_path = potential_target
        
        # Process image
        logger.info(f"Processing {i+1}/{len(input_files)}: {img_file}")
        
        # Load and preprocess input image
        _, tensor = preprocess_image(input_path)
        tensor = tensor.to(device)
        
        # Process target image if provided
        target_tensor = None
        if target_path:
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
                if metric_name not in metrics_sum:
                    metrics_sum[metric_name] = 0.0
                metrics_sum[metric_name] += metric_value
            metrics_count += 1
        
        # Convert output tensor to image and save
        output_image = postprocess_tensor(output_tensor)
        output_image.save(output_path)
        
        # Save comparisons if requested
        if save_visualizations and target_tensor is not None:
            vis_dir = os.path.join(output_dir, "comparisons")
            os.makedirs(vis_dir, exist_ok=True)
            
            # Create comparison visualization
            plt.figure(figsize=(12, 4))
            
            # Input
            plt.subplot(1, 3, 1)
            plt.title("Input (Low Quality)")
            input_img = Image.open(input_path).convert('L')
            plt.imshow(np.array(input_img), cmap='gray')
            plt.axis('off')
            
            # Output
            plt.subplot(1, 3, 2)
            plt.title("Output (Enhanced)")
            plt.imshow(np.array(output_image), cmap='gray')
            plt.axis('off')
            
            # Target
            plt.subplot(1, 3, 3)
            plt.title("Target (High Quality)")
            target_img = Image.open(target_path).convert('L')
            plt.imshow(np.array(target_img), cmap='gray')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f"compare_{img_file}"))
            plt.close()
        
        # Clean up GPU memory (important for T4 GPUs)
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Calculate and report average metrics
    if metrics_count > 0:
        logger.info("===== Average Metrics =====")
        for metric_name, metric_sum in metrics_sum.items():
            avg_value = metric_sum / metrics_count
            logger.info(f"Average {metric_name.upper()}: {avg_value:.4f}")
    
    # Report processing time
    elapsed = time.time() - start_time
    logger.info(f"Processed {len(input_files)} images in {elapsed:.2f} seconds "
               f"({len(input_files)/elapsed:.2f} images/sec)")

def main(args):
    """Main inference function."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    logger.info(f"Using device: {device}")
    
    # Print GPU info for Colab T4
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        logger.info(f"Memory cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    
    # Check if AMP can be used
    use_amp = args.use_amp and AMP_AVAILABLE and device.type == 'cuda'
    if args.use_amp and not use_amp:
        logger.warning("AMP requested but not available. Falling back to full precision.")
    if use_amp:
        logger.info("Using Automatic Mixed Precision (AMP) for inference.")
    
    try:
        # Find checkpoint
        checkpoint_path = find_best_checkpoint(args.checkpoint_dir, args.model_type)
        
        # Load model
        model = load_model(
            args.model_type, 
            checkpoint_path, 
            device,
            base_filters=args.base_filters,
            num_features=args.num_features,
            scale=args.scale,
            num_blocks=args.num_blocks,
            num_res_blocks=args.num_res_blocks
        )
        
        # Apply torch.compile if requested and available
        use_compile = args.use_compile and COMPILE_AVAILABLE and device.type == 'cuda'
        if args.use_compile and not use_compile:
            logger.warning("Model compilation requested but not available. Using standard execution.")
        if use_compile:
            logger.info("Using torch.compile to optimize model execution.")
            try:
                # For T4 GPUs on Colab, 'reduce-overhead' is safer than 'max-autotune'
                model = compile(model, mode="reduce-overhead", fullgraph=False)
                logger.info("Model successfully compiled!")
            except Exception as e:
                logger.error(f"Error compiling model: {e}. Falling back to standard execution.")
                use_compile = False
        
        # Batch mode
        if args.batch_mode:
            if os.path.isdir(args.input_image):
                process_batch(
                    model,
                    args.input_image,
                    args.output_image,
                    device,
                    args.target_image if os.path.isdir(args.target_image) else None,
                    args.save_visualizations,
                    use_amp
                )
            else:
                logger.warning("Batch mode specified but input is not a directory. Falling back to single image mode.")
                process_single_image(
                    model,
                    args.input_image,
                    args.output_image,
                    args.target_image,
                    device,
                    args.show_comparison,
                    args.show_diff,
                    use_amp
                )
        # Single image mode
        else:
            process_single_image(
                model,
                args.input_image,
                args.output_image,
                args.target_image,
                device,
                args.show_comparison,
                args.show_diff,
                use_amp
            )
    
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MRI Super-Resolution Inference")
    
    # Input/output arguments
    parser.add_argument('--input_image', type=str, required=True, 
                        help="Path to input image or directory")
    parser.add_argument('--output_image', type=str, default='output.png', 
                        help="Path for output image or directory")
    parser.add_argument('--target_image', type=str, default=None, 
                        help="Optional path to ground truth image or directory")
    
    # Model selection
    parser.add_argument('--model_type', type=str, choices=['simple', 'edsr', 'unet'], default='unet', 
                        help="Model architecture to use")
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', 
                        help="Directory containing model checkpoints")
    
    # Model parameters
    parser.add_argument('--base_filters', type=int, default=64, 
                        help="Number of base filters (for UNet)")
    parser.add_argument('--num_features', type=int, default=64,
                      help='Number of features in CNN or EDSR models')
    parser.add_argument('--scale', type=int, default=1,
                      help='Scale factor for EDSR model')
    parser.add_argument('--num_blocks', type=int, default=8,
                      help='Number of residual blocks in CNN model')
    parser.add_argument('--num_res_blocks', type=int, default=16,
                      help='Number of residual blocks in EDSR model')
    
    # Inference options
    parser.add_argument('--batch_mode', action='store_true', 
                        help="Process all images in the input directory")
    parser.add_argument('--save_visualizations', action='store_true', 
                        help="Save comparison visualizations (batch mode)")
    parser.add_argument('--show_comparison', action='store_true', 
                        help="Show comparison visualization")
    parser.add_argument('--show_diff', action='store_true', 
                        help="Show difference map in visualization")
    parser.add_argument('--cpu', action='store_true', 
                        help="Force CPU inference even if CUDA is available")
    
    # Performance options
    parser.add_argument("--use_amp", action="store_true",
                       help="Use Automatic Mixed Precision for inference on RTX GPUs")
    parser.add_argument("--use_compile", action="store_true",
                      help="Use torch.compile to optimize model execution on RTX GPUs")
    
    args = parser.parse_args()
    main(args)
