import argparse
import os
import sys
from pathlib import Path
import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Add the project root directory to the Python path.
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.cnn_model import CNNSuperRes  # simple model

def infer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Choose model and checkpoint.
    if args.model_type == "simple":
        model = CNNSuperRes().to(device)
        checkpoint_name = "cnn.pth"
    elif args.model_type == "edsr":
        from models.edsr_model import EDSRSuperRes
        model = EDSRSuperRes(scale=args.scale).to(device)
        checkpoint_name = "edsr.pth"
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_name)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    # Optional: JIT compile the model for optimized inference.
    if args.jit:
        # Use the input image dimensions for tracing.
        example_input = torch.randn(1, 1, args.input_height, args.input_width).to(device)
        model = torch.jit.trace(model, example_input)
        print("Model has been JIT compiled for optimized inference.")
    
    transform = transforms.ToTensor()
    inv_transform = transforms.ToPILImage()
    
    try:
        low_res_image = Image.open(args.input_image).convert('L')
    except Exception as e:
        raise ValueError(f"Failed to open input image: {args.input_image}. Error: {e}")
    
    # Update dimensions for JIT tracing if needed.
    if args.jit:
        args.input_width, args.input_height = low_res_image.size
    
    low_res_tensor = transform(low_res_image).unsqueeze(0).to(device, non_blocking=True)
    
    with torch.no_grad():
        output = model(low_res_tensor)
    
    output_image = inv_transform(output.squeeze(0).cpu())
    output_image.save(args.output_image)
    print(f"Saved output image to {args.output_image}")
    
    plt.imshow(output_image, cmap='gray')
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Infer high resolution MRI from low resolution input")
    parser.add_argument('--input_image', type=str, required=True, help="Path to the low resolution input image")
    parser.add_argument('--output_image', type=str, default='output.png', help="Path to save the output high resolution image")
    parser.add_argument('--model_type', type=str, choices=['simple', 'edsr'], default='simple', help="Type of CNN model to use: 'simple' or 'edsr'")
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help="Directory where model checkpoints are saved")
    parser.add_argument('--scale', type=int, default=1, help="Upscaling factor for EDSR model. Use 1 if the input and target sizes are the same.")
    parser.add_argument('--jit', action='store_true', help="Enable TorchScript JIT compilation for optimized inference")
    # Provide dummy dimensions for JIT tracing; these will be updated based on the input image.
    parser.add_argument('--input_width', type=int, default=224, help="Width of the input image for JIT tracing")
    parser.add_argument('--input_height', type=int, default=224, help="Height of the input image for JIT tracing")
    args = parser.parse_args()
    infer(args)
