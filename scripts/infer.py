import argparse
import os
import sys
from pathlib import Path
import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Add the project root directory to the Python path.
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.cnn_model import CNNSuperRes  # simple model
from utils.losses import PSNR, ssim  # Import metrics

def calculate_metrics(output_tensor, target_tensor=None):
    """Calculate image quality metrics if a target image is provided."""
    metrics = {}
    
    if target_tensor is not None:
        # Calculate PSNR
        psnr_metric = PSNR()
        psnr_value = psnr_metric(output_tensor.unsqueeze(0), target_tensor.unsqueeze(0)).item()
        metrics['psnr'] = psnr_value
        
        # Calculate SSIM
        ssim_value = ssim(
            output_tensor.unsqueeze(0), 
            target_tensor.unsqueeze(0),
            window_size=11, 
            sigma=1.5
        ).item()
        metrics['ssim'] = ssim_value
    
    return metrics

def infer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Choose model and checkpoint.
    if args.model_type == "simple":
        model = CNNSuperRes().to(device)
        checkpoint_name = "cnn.pth"
    elif args.model_type == "edsr":
        from models.edsr_model import EDSRSuperRes
        model = EDSRSuperRes(scale=args.scale).to(device)
        checkpoint_name = "edsr.pth"
    elif args.model_type == "unet":
        from models.unet_model import UNetSuperRes
        model = UNetSuperRes(base_filters=args.base_filters).to(device)
        checkpoint_name = "unet.pth"
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    # Try to load the best model first, fall back to regular checkpoint
    best_checkpoint_path = os.path.join(args.checkpoint_dir, f"best_{checkpoint_name}")
    regular_checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_name)
    
    if os.path.exists(best_checkpoint_path):
        checkpoint_path = best_checkpoint_path
        print(f"Using best model checkpoint: {checkpoint_path}")
    elif os.path.exists(regular_checkpoint_path):
        checkpoint_path = regular_checkpoint_path
        print(f"Using regular model checkpoint: {checkpoint_path}")
    else:
        raise FileNotFoundError(f"No checkpoint found for {args.model_type} model")
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    transform = transforms.ToTensor()
    inv_transform = transforms.ToPILImage()
    
    try:
        low_res_image = Image.open(args.input_image).convert('L')
    except Exception as e:
        raise ValueError(f"Failed to open input image: {args.input_image}. Error: {e}")
    
    low_res_tensor = transform(low_res_image).unsqueeze(0).to(device, non_blocking=True)
    
    # Load ground truth image if provided
    target_tensor = None
    if args.target_image:
        try:
            target_image = Image.open(args.target_image).convert('L')
            target_tensor = transform(target_image).to(device, non_blocking=True)
        except Exception as e:
            print(f"Warning: Failed to open target image: {args.target_image}. Error: {e}")
            print("Continuing without computing quality metrics.")
    
    with torch.no_grad():
        output = model(low_res_tensor)
    
    output_tensor = output.squeeze(0).cpu()
    output_image = inv_transform(output_tensor)
    output_image.save(args.output_image)
    print(f"Saved output image to {args.output_image}")
    
    # Calculate metrics if target image is provided
    if target_tensor is not None:
        metrics = calculate_metrics(output_tensor, target_tensor.cpu())
        print("\nImage Quality Metrics:")
        print(f"PSNR: {metrics['psnr']:.2f} dB")
        print(f"SSIM: {metrics['ssim']:.4f}")
    
    # Create a figure with subplots for visualization
    if args.show_comparison:
        plt.figure(figsize=(12, 4))
        
        # Plot input image
        plt.subplot(1, 3 if target_tensor is not None else 2, 1)
        plt.imshow(low_res_image, cmap='gray')
        plt.title('Input Low-Resolution')
        plt.axis('off')
        
        # Plot output image
        plt.subplot(1, 3 if target_tensor is not None else 2, 2)
        plt.imshow(output_image, cmap='gray')
        plt.title(f'Output ({args.model_type.upper()})')
        plt.axis('off')
        
        # Plot target image if available
        if target_tensor is not None:
            plt.subplot(1, 3, 3)
            plt.imshow(target_image, cmap='gray')
            plt.title('Ground Truth')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Infer high resolution MRI from low resolution input")
    parser.add_argument('--input_image', type=str, required=True, help="Path to the low resolution input image")
    parser.add_argument('--output_image', type=str, default='output.png', help="Path to save the output high resolution image")
    parser.add_argument('--target_image', type=str, default=None, help="Optional path to ground truth high resolution image for quality metrics")
    parser.add_argument('--model_type', type=str, choices=['simple', 'edsr', 'unet'], default='simple', 
                        help="Type of CNN model to use: 'simple', 'edsr', or 'unet'")
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help="Directory where model checkpoints are saved")
    parser.add_argument('--scale', type=int, default=1, help="Upscaling factor for EDSR model. Use 1 if the input and target sizes are the same.")
    parser.add_argument('--base_filters', type=int, default=64, help="Number of base filters for U-Net")
    parser.add_argument('--show_comparison', action='store_true', help="Show comparison of input, output, and target images")
    args = parser.parse_args()
    infer(args)
