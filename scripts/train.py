#!/usr/bin/env python
import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Enable cuDNN benchmarking for fixed input sizes.
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

# Add the project root directory (one level up from the scripts folder)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.dataset import MRISuperResDataset
from utils.losses import CombinedLoss, PSNR  # Custom loss with SSIM and PSNR metric
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
        "validation_split": args.validation_split,
        "patience": args.patience
    }
    print(json.dumps(params), flush=True)
    
    # Prepare dataset
    full_dataset = MRISuperResDataset(full_res_dir=args.full_res_dir, low_res_dir=args.low_res_dir)
    
    # Split dataset into training and validation sets
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * args.validation_split)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Choose the model.
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
        error_msg = {"type": "error", "message": f"Unknown model type: {args.model_type}"}
        print(json.dumps(error_msg), flush=True)
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    # Use the custom combined loss function with SSIM
    criterion = CombinedLoss(alpha=args.ssim_weight, window_size=11, sigma=1.5, val_range=1.0, device=device)
    
    # Initialize PSNR metric for evaluation
    psnr_metric = PSNR()
    
    # Set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Set up learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    print(json.dumps({"type": "info", "message": "Starting training"}), flush=True)
    
    # Set up mixed precision training.
    from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler()
    
    # Early stopping variables
    best_val_loss = float('inf')
    early_stop_counter = 0
    best_model_state = None
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        total_train_batches = len(train_loader)
        
        # Emit epoch start info.
        print(json.dumps({
            "type": "epoch_start",
            "epoch": epoch + 1,
            "total_epochs": args.epochs,
            "total_batches": total_train_batches
        }), flush=True)
        
        for batch_idx, (low, full) in enumerate(train_loader, start=1):
            # Transfer data to GPU with non-blocking calls.
            low, full = low.to(device, non_blocking=True), full.to(device, non_blocking=True)
            optimizer.zero_grad()
            with autocast():
                outputs = model(low)
                loss = criterion(outputs, full)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            
            # Emit batch progress.
            batch_msg = {
                "type": "batch_update",
                "epoch": epoch + 1,
                "batch": batch_idx,
                "total_batches": total_train_batches,
                "loss": loss.item()
            }
            print(json.dumps(batch_msg), flush=True)
        
        avg_train_loss = train_loss / total_train_batches
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_psnr = 0.0
        total_val_batches = len(val_loader)
        
        with torch.no_grad():
            for low, full in val_loader:
                low, full = low.to(device, non_blocking=True), full.to(device, non_blocking=True)
                with autocast():
                    outputs = model(low)
                    loss = criterion(outputs, full)
                    psnr = psnr_metric(outputs, full)
                val_loss += loss.item()
                val_psnr += psnr.item()
        
        avg_val_loss = val_loss / total_val_batches
        avg_val_psnr = val_psnr / total_val_batches
        
        # Update learning rate based on validation loss
        scheduler.step(avg_val_loss)
        
        elapsed_time = time.time() - start_time
        
        # Emit epoch summary.
        epoch_summary = {
            "type": "epoch_summary",
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_psnr": avg_val_psnr,
            "lr": optimizer.param_groups[0]['lr'],
            "elapsed": elapsed_time
        }
        print(json.dumps(epoch_summary), flush=True)
        
        # Check for early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            best_model_state = model.state_dict().copy()
            
            # Save the best model so far
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            best_checkpoint_path = os.path.join(args.checkpoint_dir, f"best_{checkpoint_name}")
            torch.save(best_model_state, best_checkpoint_path)
            print(json.dumps({
                "type": "info",
                "message": f"New best model saved to {best_checkpoint_path}"
            }), flush=True)
        else:
            early_stop_counter += 1
            if early_stop_counter >= args.patience:
                print(json.dumps({
                    "type": "info",
                    "message": f"Early stopping triggered after {epoch + 1} epochs"
                }), flush=True)
                break
    
    # Save the final model
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    final_checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_name)
    
    # If we have a best model from early stopping, use that
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    torch.save(model.state_dict(), final_checkpoint_path)
    
    print(json.dumps({
        "type": "info",
        "message": f"Training completed. Final model saved to {final_checkpoint_path}"
    }), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train CNN for MRI Superresolution")
    parser.add_argument('--full_res_dir', type=str, default='./training_data', help="Directory of full resolution PNG images")
    parser.add_argument('--low_res_dir', type=str, default='./training_data_1.5T', help="Directory of downsampled PNG images")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size")
    parser.add_argument('--epochs', type=int, default=50, help="Maximum number of training epochs")
    parser.add_argument('--learning_rate', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help="Directory to save model checkpoints")
    parser.add_argument('--model_type', type=str, choices=['simple', 'edsr', 'unet'], default='simple', 
                        help="Type of CNN model to use: 'simple', 'edsr', or 'unet'")
    parser.add_argument('--scale', type=int, default=2, help="Upscaling factor for EDSR model. Use 1 if the input and target sizes are the same.")
    parser.add_argument('--validation_split', type=float, default=0.2, help="Fraction of data to use for validation")
    parser.add_argument('--patience', type=int, default=10, help="Patience for early stopping")
    parser.add_argument('--ssim_weight', type=float, default=0.5, help="Weight for SSIM loss (0-1)")
    parser.add_argument('--base_filters', type=int, default=64, help="Number of base filters for U-Net")
    args = parser.parse_args()
    train(args)
