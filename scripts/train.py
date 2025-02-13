import argparse
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.dataset import MRISuperResDataset
from models.cnn_model import CNNSuperRes

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    dataset = MRISuperResDataset(full_res_dir=args.full_res_dir, low_res_dir=args.low_res_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    model = CNNSuperRes().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for low, full in dataloader:
            low, full = low.to(device), full.to(device)
            optimizer.zero_grad()
            outputs = model(low)
            loss = criterion(outputs, full)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {avg_loss:.4f}")
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.checkpoint_dir, "cnn_superres.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print("Model saved to", checkpoint_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train CNN for MRI Superresolution")
    parser.add_argument('--full_res_dir', type=str, default='./training_data',
                        help="Directory of full resolution PNG images")
    parser.add_argument('--low_res_dir', type=str, default='./training_data_1.5T',
                        help="Directory of downsampled PNG images")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--learning_rate', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help="Directory to save model checkpoints")
    args = parser.parse_args()
    train(args)
