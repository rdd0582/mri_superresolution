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

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import project modules
from utils.losses import PSNR, SSIM
from utils.preprocessing import tensor_to_numpy

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
                num_features=kwargs.get('num_features', 64)
            ).to(device)
        elif model_type == "simple":
            from models.cnn_model import CNNSuperRes
            model = CNNSuperRes(
                in_channels=1,
                out_channels=1,
                num_features=kwargs.get('num_features', 64)
            ).to(device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
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
    np_img = np.clip((np_img + 1) / 2.0, 0, 1)  # Denormalize from [-1, 1] to [0, 1]
    
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
        # Calculate PSNR
        psnr_metric = PSNR(max_val=1.0, data_range='normalized')
        metrics['psnr'] = psnr_metric(output_tensor, target_tensor).item()
        
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

def process_single_image(model, input_path, output_path, target_path=None, device=None, show_comparison=False, show_diff=False):
    """Process a single image through the model."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load and preprocess input image
    input_image, input_tensor = preprocess_image(input_path)
    input_tensor = input_tensor.to(device)
    
    # Load target image if provided
    target_image = None
    target_tensor = None
    if target_path:
        try:
            target_image, target_tensor = preprocess_image(target_path)
            target_tensor = target_tensor.to(device)
        except Exception as e:
            logger.warning(f"Error loading target image: {e}")
    
    # Run inference
    with torch.no_grad():
        start_time = time.time()
        output_tensor = model(input_tensor)
        inference_time = time.time() - start_time
        logger.info(f"Inference completed in {inference_time:.4f} seconds")
    
    # Convert output tensor to image
    output_image = postprocess_tensor(output_tensor)
    
    # Save output image
    output_image.save(output_path)
    logger.info(f"Saved output image to {output_path}")
    
    # Calculate metrics if target is available
    metrics = None
    if target_tensor is not None:
        metrics = calculate_metrics(output_tensor.cpu(), target_tensor.cpu())
        metrics['inference_time'] = inference_time
        
        # Log metrics
        logger.info("\nImage Quality Metrics:")
        for k, v in metrics.items():
            logger.info(f"{k.upper()}: {v:.4f}")
    
    # Visualize results if requested
    if show_comparison:
        visualize_results(
            input_image, 
            output_image, 
            target_image, 
            metrics, 
            show_diff=show_diff
        )
    
    return output_image, metrics

def process_batch(model, input_dir, output_dir, device=None, target_dir=None, save_viz=False):
    """Process all images in a directory."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(input_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
    image_files.sort()
    
    if not image_files:
        logger.warning(f"No image files found in {input_dir}")
        return
    
    logger.info(f"Processing {len(image_files)} images...")
    
    # Process each image
    all_metrics = {}
    for img_file in image_files:
        input_path = os.path.join(input_dir, img_file)
        output_path = os.path.join(output_dir, f"sr_{img_file}")
        
        # Check for target image
        target_path = None
        if target_dir and os.path.exists(os.path.join(target_dir, img_file)):
            target_path = os.path.join(target_dir, img_file)
        
        # Process image
        try:
            _, metrics = process_single_image(
                model, 
                input_path, 
                output_path, 
                target_path, 
                device, 
                show_comparison=False
            )
            
            # Save metrics
            if metrics:
                all_metrics[img_file] = metrics
            
            # Create visualization if requested
            if save_viz and target_path:
                viz_path = os.path.join(output_dir, f"viz_{img_file}")
                input_image, _ = preprocess_image(input_path)
                output_image = Image.open(output_path)
                target_image, _ = preprocess_image(target_path)
                
                visualize_results(
                    input_image,
                    output_image,
                    target_image,
                    metrics,
                    show_diff=True,
                    save_path=viz_path
                )
        
        except Exception as e:
            logger.error(f"Error processing {img_file}: {e}")
    
    # Save all metrics to JSON
    if all_metrics:
        metrics_path = os.path.join(output_dir, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        logger.info(f"Saved metrics to {metrics_path}")
    
    logger.info(f"Batch processing complete. Results saved to {output_dir}")

def main(args):
    """Main inference function."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    logger.info(f"Using device: {device}")
    
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
            scale=args.scale
        )
        
        # Batch mode
        if args.batch_mode:
            if os.path.isdir(args.input_image):
                process_batch(
                    model,
                    args.input_image,
                    args.output_image,
                    device,
                    args.target_image if os.path.isdir(args.target_image) else None,
                    args.save_visualizations
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
                    args.show_diff
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
                args.show_diff
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
                        help="Number of features (for CNN/EDSR)")
    parser.add_argument('--scale', type=int, default=1, 
                        help="Upscaling factor (usually 1 for same-resolution enhancement)")
    
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
    
    args = parser.parse_args()
    main(args)
