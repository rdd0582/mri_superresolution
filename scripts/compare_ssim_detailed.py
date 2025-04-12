#!/usr/bin/env python
import os
import argparse
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

def load_best_models(weight_dirs):
    """
    Load the best model checkpoints for each SSIM weight
    
    Args:
        weight_dirs: Dictionary mapping SSIM weights to their output directories
    
    Returns:
        Dictionary mapping SSIM weights to loaded model states
    """
    models = {}
    
    for weight, dir_path in weight_dirs.items():
        # Find the best model checkpoint
        checkpoint_path = os.path.join(dir_path, f'best_model_*.pth')
        checkpoint_files = glob.glob(checkpoint_path)
        
        if not checkpoint_files:
            print(f"Warning: No checkpoint found for SSIM weight {weight}")
            continue
            
        # Load the checkpoint
        checkpoint_path = checkpoint_files[0]
        try:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            models[weight] = checkpoint
            print(f"Loaded model for SSIM weight {weight} from {checkpoint_path}")
        except Exception as e:
            print(f"Error loading checkpoint for SSIM weight {weight}: {e}")
    
    return models

def create_detailed_comparison(weight_dirs, test_image_dir, output_dir, model_type="unet"):
    """
    Create a detailed comparison of the models with different SSIM weights
    
    Args:
        weight_dirs: Dictionary mapping SSIM weights to their output directories
        test_image_dir: Directory containing test low-resolution images
        output_dir: Directory to save the comparison results
        model_type: Type of model used for training
    """
    import sys
    import os
    
    # Add project root to path to import our model
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Import the appropriate model
    if model_type == "unet":
        from models.unet_model import UNetSuperRes
        model_class = UNetSuperRes
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load models
    models = {}
    for weight, dir_path in weight_dirs.items():
        # Find the best model checkpoint
        checkpoint_path = os.path.join(dir_path, f'best_model_{model_type}.pth')
        
        if not os.path.exists(checkpoint_path):
            print(f"Warning: No checkpoint found for SSIM weight {weight} at {checkpoint_path}")
            continue
            
        # Load the checkpoint
        try:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            
            # Create model instance based on model type
            if model_type == "unet":
                model = model_class(in_channels=1, out_channels=1)
            else:
                 # This case should not be reached now
                 raise ValueError(f"Unsupported model type in loading loop: {model_type}")
            
            # Load the model weights
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            models[weight] = model
            print(f"Loaded model for SSIM weight {weight} from {checkpoint_path}")
        except Exception as e:
            print(f"Error loading checkpoint for SSIM weight {weight}: {e}")
    
    # Get test images
    test_images = glob.glob(os.path.join(test_image_dir, "*.png")) + \
                 glob.glob(os.path.join(test_image_dir, "*.jpg")) + \
                 glob.glob(os.path.join(test_image_dir, "*.tif"))
    
    if not test_images:
        print(f"No test images found in {test_image_dir}")
        return
    
    # Use a few test images for comparison
    test_images = test_images[:5]  # Limit to 5 images for clarity
    
    # Create transform to convert images to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each test image
    for img_path in test_images:
        img_name = os.path.basename(img_path)
        print(f"Processing test image: {img_name}")
        
        # Create a subdirectory for this image
        img_output_dir = os.path.join(output_dir, os.path.splitext(img_name)[0])
        os.makedirs(img_output_dir, exist_ok=True)
        
        # Load and preprocess the image
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        input_tensor = transform(img).unsqueeze(0)  # Add batch dimension
        
        # Save the original image at full resolution
        original_save_path = os.path.join(img_output_dir, "original.png")
        img.save(original_save_path, format="PNG")
        
        # Create figure for this test image (small comparison view)
        n_weights = len(models)
        if n_weights == 0:
            print("No models loaded, skipping comparison")
            return
            
        fig, axes = plt.subplots(1, n_weights + 1, figsize=(5 * (n_weights + 1), 5))
        
        # Plot the original image
        axes[0].imshow(np.array(img), cmap='gray')
        axes[0].set_title("Original Low-Res")
        axes[0].axis('off')
        
        # Process with each model
        for i, (weight, model) in enumerate(sorted(models.items())):
            with torch.no_grad():
                output = model(input_tensor)
            
            # Convert output to image
            output_img = output.squeeze(0).squeeze(0).cpu().numpy()
            
            # Save full-resolution output image
            # Convert to PIL Image (ensure proper scaling to 0-255 range)
            output_pil = Image.fromarray((output_img * 255).astype(np.uint8))
            output_save_path = os.path.join(img_output_dir, f"weight_{weight}.png")
            output_pil.save(output_save_path, format="PNG")
            
            # Plot the result for comparison figure
            axes[i+1].imshow(output_img, cmap='gray')
            axes[i+1].set_title(f"SSIM Weight: {weight}")
            axes[i+1].axis('off')
        
        # Save the comparison figure (lower resolution is fine for overview)
        plt.tight_layout()
        plt.savefig(os.path.join(img_output_dir, "comparison.png"), dpi=150)
        plt.close(fig)
    
    print(f"Detailed comparison saved to {output_dir}")
    print(f"Individual full-resolution images saved in subdirectories for each test image")

def main():
    parser = argparse.ArgumentParser(description="Create detailed comparison of MRI Super-resolution with different SSIM weights")
    
    # Input directories
    parser.add_argument('--weight_dirs', type=str, required=True,
                      help='Base directory containing subdirectories for each SSIM weight')
    parser.add_argument('--test_image_dir', type=str, required=True,
                      help='Directory containing test low-resolution images')
    
    # Model type
    parser.add_argument('--model_type', type=str, choices=['unet'], default='unet',
                      help='Model architecture used for training (only unet supported)')
    
    # Output directory
    parser.add_argument('--output_dir', type=str, default='./ssim_detailed_comparison',
                      help='Directory to save comparison results')
    
    args = parser.parse_args()
    
    # Find weight directories
    weight_dirs = {}
    for dirname in os.listdir(args.weight_dirs):
        if dirname.startswith("ssim_weight_"):
            try:
                weight = float(dirname.replace("ssim_weight_", ""))
                weight_dirs[weight] = os.path.join(args.weight_dirs, dirname)
            except ValueError:
                continue
    
    if not weight_dirs:
        print(f"No weight directories found in {args.weight_dirs}")
        return
    
    print(f"Found {len(weight_dirs)} weight directories: {sorted(weight_dirs.keys())}")
    
    # Create the detailed comparison
    create_detailed_comparison(
        weight_dirs, 
        args.test_image_dir, 
        args.output_dir,
        args.model_type
    )

if __name__ == "__main__":
    main() 