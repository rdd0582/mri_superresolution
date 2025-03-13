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
import cv2
from typing import Dict, Optional, Tuple, List, Union
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
from utils.losses import PSNR, ssim, MS_SSIM
from utils.preprocessing import (
    preprocess_slice, 
    ResizeMethod, 
    InterpolationMethod,
    tensor_to_numpy,
    numpy_to_tensor
)

def load_model(model_type: str, checkpoint_path: str, device: torch.device, **model_args) -> torch.nn.Module:
    """
    Load a model with the specified architecture and weights.
    
    Args:
        model_type: Type of model ('simple', 'edsr', 'unet')
        checkpoint_path: Path to the model checkpoint
        device: Device to load the model on
        **model_args: Additional arguments for model initialization
        
    Returns:
        Loaded model
    """
    try:
        if model_type == "simple":
            from models.cnn_model import CNNSuperRes
            model = CNNSuperRes(
                in_channels=model_args.get('in_channels', 1),
                out_channels=model_args.get('out_channels', 1),
                num_features=model_args.get('num_features', 64),
                num_blocks=model_args.get('num_blocks', 8),
                scale_factor=model_args.get('scale_factor', 1)
            ).to(device)
        elif model_type == "edsr":
            from models.edsr_model import EDSRSuperRes
            model = EDSRSuperRes(
                in_channels=model_args.get('in_channels', 1),
                out_channels=model_args.get('out_channels', 1),
                scale=model_args.get('scale', 1),
                num_res_blocks=model_args.get('num_res_blocks', 16),
                num_features=model_args.get('num_features', 64),
                res_scale=model_args.get('res_scale', 0.1),
                use_mean_shift=model_args.get('use_mean_shift', False)
            ).to(device)
        elif model_type == "unet":
            from models.unet_model import UNetSuperRes
            model = UNetSuperRes(
                in_channels=model_args.get('in_channels', 1),
                out_channels=model_args.get('out_channels', 1),
                bilinear=model_args.get('bilinear', True),
                base_filters=model_args.get('base_filters', 64),
                depth=model_args.get('depth', 4),
                norm_type=model_args.get('norm_type', 'batch'),
                use_attention=model_args.get('use_attention', True),
                scale_factor=model_args.get('scale_factor', 1),
                residual_mode=model_args.get('residual_mode', 'add')
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
        
        model.eval()
        return model
    
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def find_checkpoint(checkpoint_dir: str, model_type: str) -> str:
    """
    Find the best available checkpoint for the specified model type.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        model_type: Type of model ('simple', 'edsr', 'unet')
        
    Returns:
        Path to the checkpoint file
    """
    checkpoint_map = {
        'simple': 'cnn.pth',
        'edsr': 'edsr.pth',
        'unet': 'unet.pth'
    }
    
    if model_type not in checkpoint_map:
        raise ValueError(f"Unknown model type: {model_type}")
    
    checkpoint_name = checkpoint_map[model_type]
    best_checkpoint_path = os.path.join(checkpoint_dir, f"best_{checkpoint_name}")
    regular_checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    
    if os.path.exists(best_checkpoint_path):
        logger.info(f"Using best model checkpoint: {best_checkpoint_path}")
        return best_checkpoint_path
    elif os.path.exists(regular_checkpoint_path):
        logger.info(f"Using regular model checkpoint: {regular_checkpoint_path}")
        return regular_checkpoint_path
    else:
        raise FileNotFoundError(f"No checkpoint found for {model_type} model in {checkpoint_dir}")

def load_and_preprocess_image(image_path: str, preprocess_params: Optional[Dict] = None) -> Tuple[Image.Image, torch.Tensor]:
    """
    Load and preprocess an image.
    
    Args:
        image_path: Path to the image file
        preprocess_params: Parameters for preprocessing
        
    Returns:
        Tuple of (original PIL image, preprocessed tensor)
    """
    try:
        # Load image
        image = Image.open(image_path).convert('L')
        
        # Convert to numpy array
        image_np = np.array(image)
        
        # Apply preprocessing
        if preprocess_params is not None:
            processed_np = preprocess_slice(image_np, **preprocess_params)
        else:
            # Default preprocessing
            processed_np = preprocess_slice(
                image_np,
                normalize=True,
                to_uint8=False
            )
        
        # Convert to tensor
        if processed_np.dtype == np.uint8:
            processed_np = processed_np.astype(np.float32) / 255.0
        
        tensor = torch.from_numpy(processed_np).unsqueeze(0)  # Add channel dimension
        
        return image, tensor
    
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        raise

def calculate_metrics(output_tensor: torch.Tensor, target_tensor: torch.Tensor) -> Dict[str, float]:
    """
    Calculate image quality metrics.
    
    Args:
        output_tensor: Output image tensor
        target_tensor: Target image tensor
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Ensure tensors have batch dimension
    if output_tensor.dim() == 3:
        output_tensor = output_tensor.unsqueeze(0)
    if target_tensor.dim() == 3:
        target_tensor = target_tensor.unsqueeze(0)
    
    try:
        # Calculate PSNR
        psnr_metric = PSNR()
        psnr_value = psnr_metric(output_tensor, target_tensor).item()
        metrics['psnr'] = psnr_value
        
        # Calculate SSIM
        ssim_value = ssim(
            output_tensor, 
            target_tensor,
            window_size=11, 
            sigma=1.5
        ).item()
        metrics['ssim'] = ssim_value
        
        # Calculate MS-SSIM if images are large enough
        min_size = min(output_tensor.size(2), output_tensor.size(3))
        if min_size >= 32:  # MS-SSIM needs sufficient size
            try:
                ms_ssim_metric = MS_SSIM(window_size=11, sigma=1.5)
                ms_ssim_value = ms_ssim_metric(output_tensor, target_tensor).item()
                metrics['ms_ssim'] = ms_ssim_value
            except Exception as e:
                logger.warning(f"Error calculating MS-SSIM: {e}")
        
        # Calculate RMSE
        mse = torch.nn.functional.mse_loss(output_tensor, target_tensor).item()
        metrics['rmse'] = np.sqrt(mse)
        
        # Calculate MAE
        mae = torch.nn.functional.l1_loss(output_tensor, target_tensor).item()
        metrics['mae'] = mae
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
    
    return metrics

def visualize_results(input_image: Image.Image, 
                      output_image: Image.Image, 
                      target_image: Optional[Image.Image] = None,
                      metrics: Optional[Dict[str, float]] = None,
                      diff_map: bool = False,
                      save_path: Optional[str] = None) -> None:
    """
    Visualize the results with optional difference map.
    
    Args:
        input_image: Input low-resolution image
        output_image: Output super-resolution image
        target_image: Optional ground truth image
        metrics: Optional dictionary of quality metrics
        diff_map: Whether to show difference map
        save_path: Optional path to save the visualization
    """
    has_target = target_image is not None
    has_diff = has_target and diff_map
    
    n_cols = 2 + int(has_target) + int(has_diff)
    fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 4, 5))
    
    # Plot input image
    axes[0].imshow(input_image, cmap='gray')
    axes[0].set_title('Input Low-Resolution')
    axes[0].axis('off')
    
    # Plot output image
    axes[1].imshow(output_image, cmap='gray')
    axes[1].set_title('Super-Resolution Output')
    axes[1].axis('off')
    
    # Plot target image if available
    if has_target:
        axes[2].imshow(target_image, cmap='gray')
        axes[2].set_title('Ground Truth')
        axes[2].axis('off')
    
    # Plot difference map if requested
    if has_diff:
        # Convert images to numpy arrays
        output_np = np.array(output_image)
        target_np = np.array(target_image)
        
        # Normalize to [0, 1]
        if output_np.max() > 1:
            output_np = output_np / 255.0
        if target_np.max() > 1:
            target_np = target_np / 255.0
        
        # Calculate absolute difference
        diff = np.abs(output_np - target_np)
        
        # Plot difference map with heatmap
        diff_idx = 3
        im = axes[diff_idx].imshow(diff, cmap='hot', vmin=0, vmax=0.5)
        axes[diff_idx].set_title('Absolute Difference')
        axes[diff_idx].axis('off')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[diff_idx], fraction=0.046, pad=0.04)
    
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

def batch_inference(model: torch.nn.Module, 
                   image_paths: List[str], 
                   output_dir: str,
                   device: torch.device,
                   preprocess_params: Optional[Dict] = None,
                   target_paths: Optional[List[str]] = None,
                   save_visualizations: bool = False) -> Dict[str, Dict[str, float]]:
    """
    Run inference on a batch of images.
    
    Args:
        model: Model to use for inference
        image_paths: List of paths to input images
        output_dir: Directory to save output images
        device: Device to run inference on
        preprocess_params: Parameters for preprocessing
        target_paths: Optional list of paths to target images
        save_visualizations: Whether to save visualizations
        
    Returns:
        Dictionary of metrics for each image
    """
    os.makedirs(output_dir, exist_ok=True)
    
    all_metrics = {}
    
    for i, image_path in enumerate(image_paths):
        try:
            # Get image filename
            image_name = os.path.basename(image_path)
            output_path = os.path.join(output_dir, f"sr_{image_name}")
            
            # Load and preprocess input image
            input_image, input_tensor = load_and_preprocess_image(image_path, preprocess_params)
            
            # Load target image if available
            target_image = None
            target_tensor = None
            if target_paths and i < len(target_paths) and target_paths[i]:
                try:
                    target_image, target_tensor = load_and_preprocess_image(target_paths[i], preprocess_params)
                except Exception as e:
                    logger.warning(f"Error loading target image {target_paths[i]}: {e}")
            
            # Run inference
            input_tensor = input_tensor.to(device)
            with torch.no_grad():
                start_time = time.time()
                output_tensor = model(input_tensor.unsqueeze(0))
                inference_time = time.time() - start_time
            
            # Convert output tensor to image
            output_tensor = output_tensor.squeeze(0).cpu()
            output_np = tensor_to_numpy(output_tensor)
            
            # Ensure output is in [0, 1] range
            if output_np.max() > 1.0:
                output_np = output_np / 255.0
            
            # Convert to uint8 for saving
            output_uint8 = (output_np * 255).astype(np.uint8)
            
            # Save output image
            output_image = Image.fromarray(output_uint8.squeeze())
            output_image.save(output_path)
            logger.info(f"Saved output image to {output_path}")
            
            # Calculate metrics if target is available
            metrics = {}
            if target_tensor is not None:
                metrics = calculate_metrics(output_tensor, target_tensor.cpu())
                metrics['inference_time'] = inference_time
                all_metrics[image_name] = metrics
                
                # Log metrics
                metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
                logger.info(f"Metrics for {image_name}: {metrics_str}")
            
            # Save visualization if requested
            if save_visualizations:
                viz_path = os.path.join(output_dir, f"viz_{image_name}")
                visualize_results(
                    input_image, 
                    output_image, 
                    target_image, 
                    metrics, 
                    diff_map=True,
                    save_path=viz_path
                )
        
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
    
    # Save all metrics to JSON
    if all_metrics:
        metrics_path = os.path.join(output_dir, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        logger.info(f"Saved metrics to {metrics_path}")
    
    return all_metrics

def infer(args):
    """
    Main inference function.
    
    Args:
        args: Command-line arguments
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    logger.info(f"Using device: {device}")
    
    try:
        # Find checkpoint
        checkpoint_path = find_checkpoint(args.checkpoint_dir, args.model_type)
        
        # Prepare model arguments
        model_args = {
            'in_channels': 1,
            'out_channels': 1
        }
        
        # Add model-specific arguments
        if args.model_type == 'simple':
            model_args.update({
                'num_features': args.num_features,
                'num_blocks': args.num_blocks,
                'scale_factor': args.scale
            })
        elif args.model_type == 'edsr':
            model_args.update({
                'scale': args.scale,
                'num_res_blocks': args.num_res_blocks,
                'num_features': args.num_features,
                'res_scale': args.res_scale,
                'use_mean_shift': args.use_mean_shift
            })
        elif args.model_type == 'unet':
            model_args.update({
                'base_filters': args.base_filters,
                'depth': args.depth,
                'norm_type': args.norm_type,
                'use_attention': args.use_attention,
                'scale_factor': args.scale,
                'residual_mode': args.residual_mode
            })
        
        # Load model
        model = load_model(args.model_type, checkpoint_path, device, **model_args)
        
        # Prepare preprocessing parameters
        preprocess_params = {
            'normalize': True,
            'to_uint8': False,
            'resize_method': ResizeMethod.LETTERBOX if args.resize_method == 'letterbox' else 
                            ResizeMethod.CROP if args.resize_method == 'crop' else
                            ResizeMethod.STRETCH if args.resize_method == 'stretch' else
                            ResizeMethod.PAD
        }
        
        # Handle batch inference
        if args.batch_mode:
            # Check if input is a directory
            if os.path.isdir(args.input_image):
                # Get all image files in directory
                image_paths = [
                    os.path.join(args.input_image, f) 
                    for f in os.listdir(args.input_image) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
                ]
                
                # Sort for deterministic order
                image_paths.sort()
                
                # Get target paths if target directory is provided
                target_paths = None
                if args.target_image and os.path.isdir(args.target_image):
                    target_dir = args.target_image
                    # Match target files with input files
                    target_paths = []
                    for img_path in image_paths:
                        img_name = os.path.basename(img_path)
                        target_path = os.path.join(target_dir, img_name)
                        if os.path.exists(target_path):
                            target_paths.append(target_path)
                        else:
                            target_paths.append(None)
                            logger.warning(f"No matching target found for {img_name}")
                
                # Create output directory
                output_dir = args.output_image
                os.makedirs(output_dir, exist_ok=True)
                
                # Run batch inference
                batch_inference(
                    model, 
                    image_paths, 
                    output_dir, 
                    device,
                    preprocess_params,
                    target_paths,
                    save_visualizations=args.save_visualizations
                )
                
                logger.info(f"Batch inference completed. Results saved to {output_dir}")
                return
            else:
                logger.warning("Batch mode specified but input is not a directory. Falling back to single image mode.")
        
        # Single image inference
        # Load and preprocess input image
        input_image, input_tensor = load_and_preprocess_image(args.input_image, preprocess_params)
        
        # Load target image if provided
        target_image = None
        target_tensor = None
        if args.target_image:
            try:
                target_image, target_tensor = load_and_preprocess_image(args.target_image, preprocess_params)
            except Exception as e:
                logger.warning(f"Error loading target image: {e}")
        
        # Run inference
        input_tensor = input_tensor.to(device)
        with torch.no_grad():
            start_time = time.time()
            output_tensor = model(input_tensor.unsqueeze(0))
            inference_time = time.time() - start_time
            logger.info(f"Inference completed in {inference_time:.4f} seconds")
        
        # Convert output tensor to image
        output_tensor = output_tensor.squeeze(0).cpu()
        output_np = tensor_to_numpy(output_tensor)
        
        # Ensure output is in [0, 1] range
        if output_np.max() > 1.0:
            output_np = output_np / 255.0
        
        # Convert to uint8 for saving
        output_uint8 = (output_np * 255).astype(np.uint8)
        
        # Save output image
        output_image = Image.fromarray(output_uint8.squeeze())
        output_image.save(args.output_image)
        logger.info(f"Saved output image to {args.output_image}")
        
        # Calculate metrics if target is available
        metrics = None
        if target_tensor is not None:
            metrics = calculate_metrics(output_tensor, target_tensor.cpu())
            metrics['inference_time'] = inference_time
            
            # Log metrics
            logger.info("\nImage Quality Metrics:")
            for k, v in metrics.items():
                logger.info(f"{k.upper()}: {v:.4f}")
        
        # Visualize results
        if args.show_comparison:
            visualize_results(
                input_image, 
                output_image, 
                target_image, 
                metrics, 
                diff_map=args.show_diff
            )
    
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Enhanced inference for MRI super-resolution")
    
    # Input/output arguments
    parser.add_argument('--input_image', type=str, required=True, 
                        help="Path to the low resolution input image or directory of images")
    parser.add_argument('--output_image', type=str, default='output.png', 
                        help="Path to save the output image or directory for batch mode")
    parser.add_argument('--target_image', type=str, default=None, 
                        help="Optional path to ground truth high resolution image or directory")
    
    # Model selection and parameters
    parser.add_argument('--model_type', type=str, choices=['simple', 'edsr', 'unet'], default='simple', 
                        help="Type of model to use: 'simple', 'edsr', or 'unet'")
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', 
                        help="Directory where model checkpoints are saved")
    
    # Common model parameters
    parser.add_argument('--scale', type=int, default=1, 
                        help="Upscaling factor. Use 1 if the input and target sizes are the same.")
    parser.add_argument('--num_features', type=int, default=64, 
                        help="Number of feature channels in hidden layers")
    
    # Simple CNN parameters
    parser.add_argument('--num_blocks', type=int, default=8, 
                        help="Number of residual blocks for simple CNN model")
    
    # EDSR parameters
    parser.add_argument('--num_res_blocks', type=int, default=16, 
                        help="Number of residual blocks for EDSR model")
    parser.add_argument('--res_scale', type=float, default=0.1, 
                        help="Residual scaling factor for EDSR model")
    parser.add_argument('--use_mean_shift', action='store_true', 
                        help="Use mean shift in EDSR model")
    
    # U-Net parameters
    parser.add_argument('--base_filters', type=int, default=64, 
                        help="Number of base filters for U-Net")
    parser.add_argument('--depth', type=int, default=4, 
                        help="Depth of U-Net (number of downsampling operations)")
    parser.add_argument('--norm_type', type=str, choices=['batch', 'instance', 'group'], default='batch', 
                        help="Normalization type for U-Net")
    parser.add_argument('--use_attention', action='store_true', 
                        help="Use attention mechanisms in U-Net")
    parser.add_argument('--residual_mode', type=str, choices=['add', 'concat', 'none'], default='add', 
                        help="How to handle the global residual connection in U-Net")
    
    # Preprocessing parameters
    parser.add_argument('--resize_method', type=str, choices=['letterbox', 'crop', 'stretch', 'pad'], 
                        default='letterbox', help="Method to use for resizing images")
    
    # Inference options
    parser.add_argument('--batch_mode', action='store_true', 
                        help="Process all images in the input directory")
    parser.add_argument('--save_visualizations', action='store_true', 
                        help="Save visualizations of results")
    parser.add_argument('--show_comparison', action='store_true', 
                        help="Show comparison of input, output, and target images")
    parser.add_argument('--show_diff', action='store_true', 
                        help="Show difference map between output and target")
    parser.add_argument('--cpu', action='store_true', 
                        help="Force CPU inference even if CUDA is available")
    
    args = parser.parse_args()
    infer(args)
