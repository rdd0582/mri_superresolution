import argparse
import os
import sys
from pathlib import Path
import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.cnn_model import CNNSuperRes

def infer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNSuperRes().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    transform = transforms.ToTensor()
    inv_transform = transforms.ToPILImage()
    
    low_res_image = Image.open(args.input_image).convert('L')
    low_res_tensor = transform(low_res_image).unsqueeze(0).to(device)
    
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
    parser.add_argument('--model_path', type=str, default='./checkpoints/cnn_superres.pth',
                        help="Path to the trained model checkpoint")
    parser.add_argument('--input_image', type=str, required=True,
                        help="Path to the low resolution input image")
    parser.add_argument('--output_image', type=str, default='output.png',
                        help="Path to save the output high resolution image")
    args = parser.parse_args()
    infer(args)
