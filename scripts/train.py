#!/usr/bin/env python
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Enable cuDNN benchmarking for fixed input sizes.
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

# Add the project root directory (one level up from the scripts folder)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.dataset import MRISuperResDataset
from utils.losses import CombinedLoss  # Custom loss with cached SSIM window
from models.cnn_model import CNNSuperRes  # Simple CNN model

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Log device info.
    print(json.dumps({"type": "info", "message": f"Using device: {device}"}), flush=True)
    
    # Log training parameters.
    params = {
        "type": "params",
        "full_res_dir": args.full_res_dir,
        "low_res_dir": args.low_res_dir,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "model_type": args.model_type,
        "scale": args.scale,
    }
    print(json.dumps(params), flush=True)
    
    # Prepare dataset and DataLoader with GPU-friendly settings.
    dataset = MRISuperResDataset(full_res_dir=args.full_res_dir, low_res_dir=args.low_res_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,      # Adjust based on your CPU cores
        pin_memory=True
    )
    
    # Choose the model.
    if args.model_type == "simple":
        model = CNNSuperRes().to(device)
        checkpoint_name = "cnn.pth"
    elif args.model_type == "edsr":
        from models.edsr_model import EDSRSuperRes  # Import EDSR model if needed.
        model = EDSRSuperRes(scale=args.scale).to(device)
        checkpoint_name = "edsr.pth"
    else:
        error_msg = {"type": "error", "message": f"Unknown model type: {args.model_type}"}
        print(json.dumps(error_msg), flush=True)
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    # Use the custom combined loss function (with cached SSIM window)
    criterion = CombinedLoss(alpha=0.85, window_size=11, sigma=1.5, val_range=1.0, device=device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    print(json.dumps({"type": "info", "message": "Starting training"}), flush=True)
    
    # Set up mixed precision training.
    from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler()
    
    for epoch in range(args.epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        total_batches = len(dataloader)
        
        # Emit epoch start info.
        print(json.dumps({
            "type": "epoch_start",
            "epoch": epoch + 1,
            "total_epochs": args.epochs,
            "total_batches": total_batches
        }), flush=True)
        
        for batch_idx, (low, full) in enumerate(dataloader, start=1):
            # Transfer data to GPU with non-blocking calls.
            low, full = low.to(device, non_blocking=True), full.to(device, non_blocking=True)
            optimizer.zero_grad()
            with autocast():
                outputs = model(low)
                loss = criterion(outputs, full)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            
            # Emit batch progress.
            batch_msg = {
                "type": "batch_update",
                "epoch": epoch + 1,
                "batch": batch_idx,
                "total_batches": total_batches,
                "loss": loss.item()
            }
            print(json.dumps(batch_msg), flush=True)
        
        avg_loss = running_loss / total_batches
        elapsed_time = time.time() - start_time
        
        # Emit epoch summary.
        epoch_summary = {
            "type": "epoch_summary",
            "epoch": epoch + 1,
            "avg_loss": avg_loss,
            "elapsed": elapsed_time
        }
        print(json.dumps(epoch_summary), flush=True)
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_name)
    torch.save(model.state_dict(), checkpoint_path)
    
    print(json.dumps({
        "type": "info",
        "message": f"Training completed. Model saved to {checkpoint_path}"
    }), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train CNN for MRI Superresolution")
    parser.add_argument('--full_res_dir', type=str, default='./training_data', help="Directory of full resolution PNG images")
    parser.add_argument('--low_res_dir', type=str, default='./training_data_1.5T', help="Directory of downsampled PNG images")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--learning_rate', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help="Directory to save model checkpoints")
    parser.add_argument('--model_type', type=str, choices=['simple', 'edsr'], default='simple', help="Type of CNN model to use: 'simple' or 'edsr'")
    parser.add_argument('--scale', type=int, default=2, help="Upscaling factor for EDSR model. Use 1 if the input and target sizes are the same.")
    args = parser.parse_args()
    train(args)
