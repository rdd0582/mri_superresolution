#!/usr/bin/env python
import argparse
import json
import os
import sys
import time
import numpy as np
import logging
import random
import multiprocessing

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt

# Add the project root directory to the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import project modules
from utils.dataset import MRISuperResDataset
from utils.losses import CombinedLoss, SSIM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

# Try to import TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    logger.warning("TensorBoard not available. Install with: pip install tensorboard")

# Try to import AMP
try:
    from torch.amp import autocast, GradScaler
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False
    logger.warning("Automatic Mixed Precision not available. Using default precision.")

def log_message(message, message_type="info"):
    """Log a message both to the logger (human-readable) and as JSON to stdout (for UI)."""
    
    # --- Prepare and print JSON for UI ---
    if isinstance(message, dict):
        json_message = message.copy()
        # Format numeric values for JSON
        for key, value in json_message.items():
            if isinstance(value, float):
                json_message[key] = round(value, 6)
        json_message["type"] = message_type
        print(json.dumps(json_message), flush=True)
    else:
        json_message = {"type": message_type, "message": str(message)} # Ensure message is string
        print(json.dumps(json_message), flush=True)

    # --- Log human-readable message to standard logger ---
    if message_type == "batch_update":
         # Skip verbose batch updates in the standard log file/console logger
         pass 
    elif isinstance(message, dict):
        if message_type == "epoch_summary":
            log_msg = (
                f"Epoch {message['epoch']+1}/{message.get('total_epochs', '?')} | "
                f"Train Loss: {message.get('train_loss', 0):.4f} | "
                f"Train SSIM: {message.get('train_ssim', 0):.4f}"
            )
            if message.get('val_loss') != "N/A":
                log_msg += f" | Val Loss: {message.get('val_loss', 0):.4f} | Val SSIM: {message.get('val_ssim', 0):.4f}"
            log_msg += f" | Time: {message.get('elapsed', 0):.2f}s"
            logger.info(log_msg)
        elif message_type == "params":
            # Format params nicely for the logger, excluding the 'type' field itself
            params_str = ", ".join([f"{k}={v}" for k, v in message.items() if k != 'type'])
            logger.info(f"Training Parameters: {params_str}")
        pass
    else:
        pass

def save_example_images(low_res, high_res, output, epoch, save_dir):
    """Save sample images to visualize model performance"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Images are already in [0, 1] range, no need to unnormalize
    
    # Create a grid of sample images
    samples = min(4, low_res.size(0))
    
    plt.figure(figsize=(15, 5))
    
    for i in range(samples):
        # Get sample images
        low = low_res[i].cpu().squeeze(0).numpy()
        high = high_res[i].cpu().squeeze(0).numpy()
        pred = output[i].cpu().squeeze(0).numpy()
        
        # Plot images
        plt.subplot(samples, 3, i*3 + 1)
        plt.imshow(low, cmap='gray')
        if i == 0:
            plt.title("Low Resolution")
        plt.axis('off')
        
        plt.subplot(samples, 3, i*3 + 2)
        plt.imshow(pred, cmap='gray')
        if i == 0:
            plt.title("Generated")
        plt.axis('off')
        
        plt.subplot(samples, 3, i*3 + 3)
        plt.imshow(high, cmap='gray')
        if i == 0:
            plt.title("High Resolution")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'comparison_epoch_{epoch}.png'), dpi=150)
    plt.close()

def get_recommended_workers():
    """Get recommended number of workers based on system CPU count"""
    try:
        # Use CPU count as the recommendation with a reasonable upper limit
        return min(multiprocessing.cpu_count(), 16)
    except:
        # If we can't determine CPU count, use a conservative default
        return 4

def train(args):
    """Train the MRI quality enhancement model"""
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        
    # Create output directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.join(args.checkpoint_dir, 'samples'), exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    log_message(f"Using device: {device}")

    # Check if AMP can be used
    use_amp = args.use_amp and AMP_AVAILABLE and device.type == 'cuda'
    if args.use_amp and not use_amp:
        log_message("AMP requested but not available. Falling back to full precision.")
    if use_amp:
        log_message("Using Automatic Mixed Precision (AMP) training.")
        scaler = GradScaler()
    
    # Create model based on model_type
    if args.model_type == "unet":
        from models.unet_model import UNetSuperRes
        model = UNetSuperRes(
            in_channels=1,
            out_channels=1,
            base_filters=args.base_filters,
            initial_alpha=args.initial_alpha
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    model = model.to(device)

    # Log memory usage *after* moving the model to the device
    if device.type == 'cuda':
        log_message(f"GPU: {torch.cuda.get_device_name(0)}")
        log_message(f"Memory allocated after model load: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        log_message(f"Memory reserved after model load: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Create scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=args.patience // 2
    )

    # Create dataset with normalization to [0, 1] range
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    dataset = MRISuperResDataset(
        full_res_dir=args.full_res_dir,
        low_res_dir=args.low_res_dir,
        transform=transform,
        augmentation=args.augmentation
    )
    
    # Split dataset
    dataset_size = len(dataset)
    val_size = int(args.validation_split * dataset_size)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None
    )
    
    # Create loss function and metrics
    criterion = CombinedLoss(
        ssim_weight=args.ssim_weight,
        perceptual_weight=args.perceptual_weight,
        vgg_layer_idx=args.vgg_layer_idx,
        perceptual_loss_type=args.perceptual_loss_type,
        window_size=11,
        sigma=1.5,
        val_range=1.0,  # For [0, 1] normalized data
        device=device
    )
    
    # Using consistent parameters for metrics
    ssim_metric = SSIM(window_size=11, sigma=1.5, val_range=1.0)
    
    # Create tensorboard writer if available
    writer = SummaryWriter(args.log_dir) if TENSORBOARD_AVAILABLE and args.use_tensorboard else None

    # Log training parameters
    log_message({
        "type": "params",
        "model_type": args.model_type,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "ssim_weight": args.ssim_weight,
        "perceptual_weight": args.perceptual_weight,
        "initial_alpha": args.initial_alpha,
        "augmentation": args.augmentation,
        "validation_split": args.validation_split,
        "patience": args.patience,
        "num_workers": args.num_workers,
        "seed": args.seed
    })

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0  # Counts consecutive validation epochs without improvement
    
    # Visualization frequency (only save images every X epochs)
    vis_frequency = max(1, args.epochs // 20)  # Approx 20 visualizations over the full training
    
    for epoch in range(args.epochs):
        # Always validate every epoch
        val_frequency = 1
        
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_ssim = 0.0
        
        # Use tqdm if available for progress bar
        try:
            from tqdm import tqdm
            train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        except ImportError:
            train_iterator = train_loader
        
        for batch_idx, (low_res, high_res) in enumerate(train_iterator):
            low_res = low_res.to(device, non_blocking=True)
            high_res = high_res.to(device, non_blocking=True)
            
            # Forward pass with or without AMP
            optimizer.zero_grad(set_to_none=True)
            
            if use_amp:
                with autocast('cuda'):
                    output = model(low_res)
                    loss = criterion(output, high_res)
                
                # Backward pass with scaler
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(low_res)
                loss = criterion(output, high_res)
                
                # Standard backward pass
                loss.backward()
                optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            with torch.no_grad():
                train_ssim += ssim_metric(output, high_res).item()
            
            # Log batch update less frequently for larger datasets
            if batch_idx % max(10, len(train_loader) // 10) == 0:
                log_message({
                    "type": "batch_update",
                    "epoch": epoch,
                    "batch": batch_idx,
                    "total_batches": len(train_loader),
                    "loss": loss.item()
                })
                
                # Update progress bar with current metrics
                if isinstance(train_iterator, tqdm):
                    current_loss = train_loss / (batch_idx + 1)
                    current_ssim = train_ssim / (batch_idx + 1)
                    train_iterator.set_postfix(
                        loss=f"{current_loss:.4f}",
                        ssim=f"{current_ssim:.4f}",
                        lr=f"{optimizer.param_groups[0]['lr']:.2e}"
                    )
        
        # Calculate average training metrics
        train_loss /= len(train_loader)
        train_ssim /= len(train_loader)
        
        # Validation phase - only run every val_frequency epochs
        if epoch % val_frequency == 0:
            model.eval()
            val_loss = 0.0
            val_ssim = 0.0
            
            with torch.no_grad():
                # Add validation progress bar
                try:
                    val_iterator = tqdm(val_loader, desc=f"Validating (Epoch {epoch+1})")
                except ImportError:
                    val_iterator = val_loader
                    
                for low_res, high_res in val_iterator:
                    low_res = low_res.to(device, non_blocking=True)
                    high_res = high_res.to(device, non_blocking=True)
                    
                    if use_amp:
                        with autocast('cuda'):
                            output = model(low_res)
                            loss = criterion(output, high_res)
                    else:
                        output = model(low_res)
                        loss = criterion(output, high_res)
                    
                    val_loss += loss.item()
                    val_ssim += ssim_metric(output, high_res).item()
                    
                    # Update validation progress bar
                    if isinstance(val_iterator, tqdm):
                        current_val_loss = val_loss / (val_iterator.n + 1)
                        current_val_ssim = val_ssim / (val_iterator.n + 1)
                        val_iterator.set_postfix(
                            loss=f"{current_val_loss:.4f}",
                            ssim=f"{current_val_ssim:.4f}"
                        )
                    
                    # Save the last batch for visualization
                    vis_low_res, vis_high_res, vis_output = low_res, high_res, output
            
            # Calculate average validation metrics
            val_loss /= len(val_loader)
            val_ssim /= len(val_loader)
            
            # Store previous learning rate to detect changes
            prev_lr = optimizer.param_groups[0]['lr']
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Check if learning rate changed
            current_lr = optimizer.param_groups[0]['lr']
            if current_lr != prev_lr:
                log_message(f"Learning rate adjusted from {prev_lr:.2e} to {current_lr:.2e}")
            
            # Check for best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save model checkpoint
                checkpoint_path = os.path.join(args.checkpoint_dir, f'best_model_{args.model_type}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_loss,
                    'val_ssim': val_ssim,
                }, checkpoint_path)
                
                log_message(f"Saved best model with validation loss: {val_loss:.6f}")
            else:
                patience_counter += 1  # Increment counter for validation epochs without improvement
        else:
            val_loss = "N/A"
            val_ssim = "N/A"
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Log epoch summary
        log_message({
            "type": "epoch_summary",
            "epoch": epoch,
            "total_epochs": args.epochs,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_ssim": train_ssim,
            "val_ssim": val_ssim,
            "elapsed": epoch_time,
            "lr": optimizer.param_groups[0]['lr']
        })
        
        # Log to tensorboard if available
        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            if val_loss != "N/A":
                writer.add_scalar('Loss/val', val_loss, epoch)
                writer.add_scalar('SSIM/val', val_ssim, epoch)
            writer.add_scalar('SSIM/train', train_ssim, epoch)
        
        # Save visualization less frequently to reduce overhead
        if epoch % vis_frequency == 0 and val_loss != "N/A":
            save_example_images(
                vis_low_res, 
                vis_high_res, 
                vis_output, 
                epoch, 
                os.path.join(args.checkpoint_dir, 'samples')
            )
        
        # Early stopping - only check when validation is performed
        if val_loss != "N/A" and patience_counter >= args.patience:
            log_message(f"Early stopping triggered after {epoch + 1} epochs")
            break
    
    # Save final model
    final_path = os.path.join(args.checkpoint_dir, f'final_model_{args.model_type}.pth')
    final_val_loss = best_val_loss if val_loss == "N/A" else val_loss
    final_val_ssim = 0.0 if val_ssim == "N/A" else val_ssim
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': final_val_loss,
        'val_ssim': final_val_ssim
    }, final_path)
    
    log_message(f"Training completed. Final model saved to {final_path}")
    
    if writer:
        writer.close()
    
    return final_path

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train MRI quality enhancement model")
    
    # Data paths
    parser.add_argument('--full_res_dir', type=str, required=True,
                      help='Directory containing high-quality MRI slices')
    parser.add_argument('--low_res_dir', type=str, required=True,
                      help='Directory containing low-quality MRI slices')
    
    # Model selection parameter
    parser.add_argument('--model_type', type=str, choices=['unet'], default='unet',
                      help='Model architecture to use (only unet is supported)')
    
    # Model-specific parameters
    parser.add_argument('--base_filters', type=int, default=32,
                      help='Number of base filters in the UNet model')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8,
                      help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                      help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                      help='Weight decay for optimizer')
    parser.add_argument('--ssim_weight', type=float, default=0.3,
                      help='Weight for SSIM loss component (0-1)')
    parser.add_argument('--perceptual_weight', type=float, default=0.0,
                        help='Weight for Perceptual loss component (0-1, set > 0 to enable)')
    parser.add_argument('--vgg_layer_idx', type=int, default=35,
                        help='VGG19 layer index for perceptual loss features (e.g., 35 for relu5_4)')
    parser.add_argument('--perceptual_loss_type', type=str, default='l1', choices=['l1', 'l2', 'mse'],
                        help='Type of distance metric for perceptual loss (l1 or l2/mse)')
    parser.add_argument('--initial_alpha', type=float, default=0.0,
                        help='Initial weight for blending bilinear and pixelshuffle outputs')
    parser.add_argument('--validation_split', type=float, default=0.2,
                      help='Fraction of data to use for validation')
    parser.add_argument('--patience', type=int, default=10,
                      help='Early stopping patience')
    parser.add_argument('--num_workers', type=int, default=get_recommended_workers(),
                      help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=random.randint(1, 10000),
                      help='Random seed for reproducibility (default: random)')
    
    # Options
    parser.add_argument('--augmentation', action='store_true',
                      help='Enable data augmentation')
    parser.add_argument('--use_tensorboard', action='store_true',
                      help='Use TensorBoard for logging')
    parser.add_argument('--use_amp', action='store_true',
                        help='Use Automatic Mixed Precision training for faster performance on RTX GPUs')
    parser.add_argument('--cpu', action='store_true',
                      help='Force using CPU even if CUDA is available')
    
    # Directories
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                      help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs',
                      help='Directory to save logs')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    train(args)
