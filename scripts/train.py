#!/usr/bin/env python
import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import logging
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR
import torch.nn.functional as F

# Enable cuDNN benchmarking for fixed input sizes
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

# Add the project root directory (one level up from the scripts folder)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import project modules
from utils.dataset import MRISuperResDataset, create_subject_aware_split, PatchDataset
from utils.losses import CombinedLoss, PSNR, MS_SSIM, ContentLoss, AdaptiveLoss, ssim
from models.cnn_model import CNNSuperRes
from models.edsr_model import EDSRSuperRes
from models.unet_model import UNetSuperRes

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

def log_message(message, message_type="info"):
    """Log a message both to the logger and as JSON to stdout for the UI."""
    if isinstance(message, dict):
        message["type"] = message_type
        print(json.dumps(message), flush=True)
        logger.info(f"{message_type}: {message}")
    else:
        print(json.dumps({"type": message_type, "message": message}), flush=True)
        logger.info(message)

def create_model(model_type, device, **kwargs):
    """Create a model based on the specified type and parameters."""
    if model_type == "simple":
        model = CNNSuperRes(
            in_channels=kwargs.get('in_channels', 1),
            out_channels=kwargs.get('out_channels', 1),
            num_features=kwargs.get('num_features', 64),
            num_blocks=kwargs.get('num_blocks', 8),
            scale_factor=kwargs.get('scale_factor', 1)
        ).to(device)
        checkpoint_name = "cnn.pth"
    elif model_type == "edsr":
        model = EDSRSuperRes(
            in_channels=kwargs.get('in_channels', 1),
            out_channels=kwargs.get('out_channels', 1),
            scale=kwargs.get('scale', 1),
            num_res_blocks=kwargs.get('num_res_blocks', 16),
            num_features=kwargs.get('num_features', 64),
            res_scale=kwargs.get('res_scale', 0.1),
            use_mean_shift=kwargs.get('use_mean_shift', False)
        ).to(device)
        checkpoint_name = "edsr.pth"
    elif model_type == "unet":
        model = UNetSuperRes(
            in_channels=kwargs.get('in_channels', 1),
            out_channels=kwargs.get('out_channels', 1),
            bilinear=kwargs.get('bilinear', True),
            base_filters=kwargs.get('base_filters', 64),
            depth=kwargs.get('depth', 4),
            norm_type=kwargs.get('norm_type', 'batch'),
            use_attention=kwargs.get('use_attention', True),
            scale_factor=kwargs.get('scale_factor', 1),
            residual_mode=kwargs.get('residual_mode', 'add')
        ).to(device)
        checkpoint_name = "unet.pth"
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model, checkpoint_name

def create_criterion(loss_type, device, **kwargs):
    """Create a loss function based on the specified type and parameters."""
    if loss_type == "combined":
        criterion = CombinedLoss(
            alpha=kwargs.get('ssim_weight', 0.5),
            window_size=kwargs.get('window_size', 11),
            sigma=kwargs.get('sigma', 1.5),
            val_range=kwargs.get('val_range', 1.0),
            device=device,
            use_ms_ssim=kwargs.get('use_ms_ssim', False),
            use_edge_loss=kwargs.get('use_edge_loss', False),
            use_freq_loss=kwargs.get('use_freq_loss', False),
            edge_weight=kwargs.get('edge_weight', 0.1),
            freq_weight=kwargs.get('freq_weight', 0.1)
        )
    elif loss_type == "l1":
        criterion = torch.nn.L1Loss()
    elif loss_type == "mse":
        criterion = torch.nn.MSELoss()
    elif loss_type == "ssim":
        criterion = lambda x, y: 1.0 - MS_SSIM()(x, y)
    elif loss_type == "content":
        criterion = ContentLoss()
    elif loss_type == "adaptive":
        # Create an adaptive loss that combines L1, MSE, and SSIM
        loss_modules = [
            torch.nn.L1Loss(),
            torch.nn.MSELoss(),
            lambda x, y: 1.0 - MS_SSIM()(x, y)
        ]
        criterion = AdaptiveLoss(loss_modules)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    return criterion

def create_optimizer(optimizer_type, model_params, **kwargs):
    """Create an optimizer based on the specified type and parameters."""
    if optimizer_type == "adam":
        optimizer = optim.Adam(
            model_params,
            lr=kwargs.get('learning_rate', 1e-3),
            weight_decay=kwargs.get('weight_decay', 0)
        )
    elif optimizer_type == "adamw":
        optimizer = optim.AdamW(
            model_params,
            lr=kwargs.get('learning_rate', 1e-3),
            weight_decay=kwargs.get('weight_decay', 1e-4)
        )
    elif optimizer_type == "sgd":
        optimizer = optim.SGD(
            model_params,
            lr=kwargs.get('learning_rate', 1e-2),
            momentum=kwargs.get('momentum', 0.9),
            weight_decay=kwargs.get('weight_decay', 0)
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    return optimizer

def create_scheduler(scheduler_type, optimizer, **kwargs):
    """Create a learning rate scheduler based on the specified type and parameters."""
    if scheduler_type == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=kwargs.get('factor', 0.5),
            patience=kwargs.get('patience', 5),
            verbose=True
        )
    elif scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get('t_max', 10),
            eta_min=kwargs.get('min_lr', 1e-6)
        )
    elif scheduler_type == "onecycle":
        scheduler = OneCycleLR(
            optimizer,
            max_lr=kwargs.get('max_lr', 1e-3),
            total_steps=kwargs.get('total_steps'),
            pct_start=kwargs.get('pct_start', 0.3),
            div_factor=kwargs.get('div_factor', 25.0),
            final_div_factor=kwargs.get('final_div_factor', 1e4)
        )
    elif scheduler_type == "none":
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    return scheduler

def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, checkpoint_path, **kwargs):
    """Save a checkpoint with model state and training metadata."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'timestamp': datetime.now().isoformat()
    }
    
    # Add scheduler state if available
    if scheduler is not None:
        try:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        except:
            pass
    
    # Add any additional metadata
    for key, value in kwargs.items():
        checkpoint[key] = value
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)
    log_message(f"Checkpoint saved to {checkpoint_path}")

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, device=None):
    """Load a checkpoint with model state and training metadata."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except:
            log_message(f"Warning: Could not load scheduler state from checkpoint", "warning")
    
    epoch = checkpoint.get('epoch', 0)
    val_loss = checkpoint.get('val_loss', float('inf'))
    
    log_message(f"Loaded checkpoint from epoch {epoch} with validation loss {val_loss:.4f}")
    
    return epoch, val_loss

def train_epoch(model, train_loader, criterion, optimizer, device, scaler, epoch, args):
    """Train the model for one epoch."""
    model.train()
    train_loss = 0.0
    total_train_batches = len(train_loader)
    
    # Log epoch start
    log_message({
        "epoch": epoch + 1,
        "total_epochs": args.epochs,
        "total_batches": total_train_batches
    }, "epoch_start")
    
    for batch_idx, (low, full) in enumerate(train_loader, start=1):
        # Transfer data to device
        low, full = low.to(device, non_blocking=True), full.to(device, non_blocking=True)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(enabled=args.mixed_precision):
            outputs = model(low)
            loss = criterion(outputs, full)
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Gradient clipping if enabled
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        # Optimizer step with scaling
        scaler.step(optimizer)
        scaler.update()
        
        # Update running loss
        train_loss += loss.item()
        
        # Log batch progress
        if batch_idx % args.log_interval == 0:
            log_message({
                "epoch": epoch + 1,
                "batch": batch_idx,
                "total_batches": total_train_batches,
                "loss": loss.item()
            }, "batch_update")
    
    # Calculate average loss
    avg_train_loss = train_loss / total_train_batches
    
    return avg_train_loss

def validate(model, val_loader, criterion, metrics, device, epoch, args):
    """Validate the model on the validation set."""
    model.eval()
    val_loss = 0.0
    metrics_values = {name: 0.0 for name in metrics.keys()}
    total_val_batches = len(val_loader)
    
    with torch.no_grad():
        for low, full in val_loader:
            # Transfer data to device
            low, full = low.to(device, non_blocking=True), full.to(device, non_blocking=True)
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=args.mixed_precision):
                outputs = model(low)
                loss = criterion(outputs, full)
            
            # Update running loss
            val_loss += loss.item()
            
            # Calculate metrics
            for name, metric_fn in metrics.items():
                metrics_values[name] += metric_fn(outputs, full).item()
    
    # Calculate averages
    avg_val_loss = val_loss / total_val_batches
    avg_metrics = {name: value / total_val_batches for name, value in metrics_values.items()}
    
    return avg_val_loss, avg_metrics

def train(args):
    """Main training function."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_message(f"Using device: {device}")
    
    # Log training parameters
    log_message({
        "full_res_dir": args.full_res_dir,
        "low_res_dir": args.low_res_dir,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "model_type": args.model_type,
        "scale": args.scale,
        "validation_split": args.validation_split,
        "patience": args.lr_patience,
        "loss_type": args.loss_type,
        "optimizer_type": args.optimizer_type,
        "scheduler_type": args.scheduler_type,
        "augmentation": args.augmentation,
        "use_patches": args.use_patches,
        "mixed_precision": args.mixed_precision
    }, "params")
    
    # Set up TensorBoard if available
    if TENSORBOARD_AVAILABLE and args.use_tensorboard:
        log_dir = os.path.join(args.log_dir, f"{args.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        writer = SummaryWriter(log_dir=log_dir)
        log_message(f"TensorBoard logs will be saved to {log_dir}")
    else:
        writer = None
    
    # Set up augmentation parameters
    augmentation_params = {
        'flip_prob': args.flip_prob,
        'rotate_prob': args.rotate_prob,
        'rotate_range': (-args.rotate_angle, args.rotate_angle),
        'brightness_prob': args.brightness_prob,
        'brightness_range': (1.0 - args.brightness_factor, 1.0 + args.brightness_factor),
        'contrast_prob': args.contrast_prob,
        'contrast_range': (1.0 - args.contrast_factor, 1.0 + args.contrast_factor),
        'noise_prob': args.noise_prob,
        'noise_std': args.noise_std
    }
    
    # Prepare dataset
    full_dataset = MRISuperResDataset(
        full_res_dir=args.full_res_dir,
        low_res_dir=args.low_res_dir,
        augmentation=args.augmentation,
        augmentation_params=augmentation_params if args.augmentation else None,
        normalize=True
    )
    
    # Split dataset
    if args.subject_aware_split:
        train_dataset, val_dataset, test_dataset = create_subject_aware_split(
            full_dataset,
            val_ratio=args.validation_split,
            test_ratio=args.test_split,
            seed=args.seed
        )
    else:
        # Traditional random split
        dataset_size = len(full_dataset)
        val_size = int(dataset_size * args.validation_split)
        test_size = int(dataset_size * args.test_split)
        train_size = dataset_size - val_size - test_size
        
        generator = torch.Generator().manual_seed(args.seed)
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size, test_size], generator=generator
        )
    
    # Use patch dataset if enabled
    if args.use_patches:
        train_dataset = PatchDataset(
            train_dataset,
            patch_size=args.patch_size,
            stride=args.patch_stride
        )
        log_message(f"Using patches of size {args.patch_size}x{args.patch_size} with stride {args.patch_stride}")
        log_message(f"Training dataset size: {len(train_dataset)} patches")
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    model_args = {
        'in_channels': 1,
        'out_channels': 1,
        'scale_factor': args.scale,
        'num_features': args.num_features,
        'num_blocks': args.num_blocks,
        'num_res_blocks': args.num_res_blocks,
        'res_scale': args.res_scale,
        'use_mean_shift': args.use_mean_shift,
        'base_filters': args.base_filters,
        'depth': args.depth,
        'norm_type': args.norm_type,
        'use_attention': args.use_attention,
        'residual_mode': args.residual_mode,
        'bilinear': args.bilinear
    }
    
    model, checkpoint_name = create_model(args.model_type, device, **model_args)
    
    # Create loss function
    loss_args = {
        'ssim_weight': args.ssim_weight,
        'window_size': args.window_size,
        'sigma': args.sigma,
        'val_range': 1.0,
        'use_ms_ssim': args.use_ms_ssim,
        'use_edge_loss': args.use_edge_loss,
        'use_freq_loss': args.use_freq_loss,
        'edge_weight': args.edge_weight,
        'freq_weight': args.freq_weight
    }
    
    criterion = create_criterion(args.loss_type, device, **loss_args)
    
    # Create metrics
    metrics = {
        'psnr': PSNR(),
        'ssim': lambda x, y: ssim(x, y, window_size=args.window_size, sigma=args.sigma)
    }
    
    # Create optimizer
    optimizer_args = {
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'momentum': args.momentum
    }
    
    optimizer = create_optimizer(args.optimizer_type, model.parameters(), **optimizer_args)
    
    # Create scheduler
    scheduler_args = {
        'factor': args.lr_factor,
        'patience': args.lr_patience,
        't_max': args.epochs,
        'min_lr': args.min_lr,
        'max_lr': args.learning_rate,
        'total_steps': len(train_loader) * args.epochs,
        'pct_start': args.warmup_pct,
        'div_factor': 25.0,
        'final_div_factor': 1e4
    }
    
    scheduler = create_scheduler(args.scheduler_type, optimizer, **scheduler_args)
    
    # Set up mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        checkpoint_path = os.path.join(args.checkpoint_dir, f"best_{checkpoint_name}")
        if os.path.exists(checkpoint_path):
            start_epoch, best_val_loss = load_checkpoint(
                checkpoint_path, model, optimizer, scheduler, device
            )
            start_epoch += 1  # Start from the next epoch
    
    # Early stopping variables
    early_stop_counter = 0
    
    # Training loop
    log_message("Starting training")
    
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        
        # Training phase
        avg_train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler, epoch, args)
        
        # Validation phase
        avg_val_loss, avg_metrics = validate(model, val_loader, criterion, metrics, device, epoch, args)
        
        # Update learning rate based on validation loss
        if args.scheduler_type == "plateau":
            scheduler.step(avg_val_loss)
        elif scheduler is not None and args.scheduler_type != "onecycle":
            scheduler.step()
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Log epoch summary
        epoch_summary = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "metrics": avg_metrics,
            "lr": optimizer.param_groups[0]['lr'],
            "elapsed": elapsed_time
        }
        log_message(epoch_summary, "epoch_summary")
        
        # Log to TensorBoard if available
        if writer is not None:
            writer.add_scalar('Loss/train', avg_train_loss, epoch)
            writer.add_scalar('Loss/val', avg_val_loss, epoch)
            for name, value in avg_metrics.items():
                writer.add_scalar(f'Metrics/{name}', value, epoch)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Check for early stopping and model saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            
            # Save the best model
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            best_checkpoint_path = os.path.join(args.checkpoint_dir, f"best_{checkpoint_name}")
            
            save_checkpoint(
                model, optimizer, scheduler, epoch, avg_val_loss, best_checkpoint_path,
                metrics=avg_metrics,
                args=vars(args)
            )
        else:
            early_stop_counter += 1
            if early_stop_counter >= args.patience:
                log_message(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        # Save regular checkpoint every checkpoint_interval epochs
        if args.checkpoint_interval > 0 and (epoch + 1) % args.checkpoint_interval == 0:
            checkpoint_path = os.path.join(
                args.checkpoint_dir, 
                f"{args.model_type}_epoch{epoch+1}_{checkpoint_name}"
            )
            save_checkpoint(
                model, optimizer, scheduler, epoch, avg_val_loss, checkpoint_path,
                metrics=avg_metrics,
                args=vars(args)
            )
    
    # Save the final model
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    final_checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_name)
    
    # Load the best model if available
    best_checkpoint_path = os.path.join(args.checkpoint_dir, f"best_{checkpoint_name}")
    if os.path.exists(best_checkpoint_path):
        load_checkpoint(best_checkpoint_path, model)
    
    # Save the final model
    torch.save(model.state_dict(), final_checkpoint_path)
    log_message(f"Training completed. Final model saved to {final_checkpoint_path}")
    
    # Close TensorBoard writer
    if writer is not None:
        writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Enhanced training for MRI Superresolution")
    
    # Dataset parameters
    parser.add_argument('--full_res_dir', type=str, default='./training_data', 
                        help="Directory of full resolution PNG images")
    parser.add_argument('--low_res_dir', type=str, default='./training_data_1.5T', 
                        help="Directory of downsampled PNG images")
    parser.add_argument('--validation_split', type=float, default=0.2, 
                        help="Fraction of data to use for validation")
    parser.add_argument('--test_split', type=float, default=0.1, 
                        help="Fraction of data to use for testing")
    parser.add_argument('--subject_aware_split', action='store_true', 
                        help="Split dataset by subject rather than randomly")
    parser.add_argument('--seed', type=int, default=42, 
                        help="Random seed for reproducibility")
    
    # Augmentation parameters
    parser.add_argument('--augmentation', action='store_true', 
                        help="Enable data augmentation")
    parser.add_argument('--flip_prob', type=float, default=0.5, 
                        help="Probability of horizontal flip")
    parser.add_argument('--rotate_prob', type=float, default=0.5, 
                        help="Probability of rotation")
    parser.add_argument('--rotate_angle', type=float, default=5.0, 
                        help="Maximum rotation angle in degrees")
    parser.add_argument('--brightness_prob', type=float, default=0.3, 
                        help="Probability of brightness adjustment")
    parser.add_argument('--brightness_factor', type=float, default=0.1, 
                        help="Maximum brightness adjustment factor")
    parser.add_argument('--contrast_prob', type=float, default=0.3, 
                        help="Probability of contrast adjustment")
    parser.add_argument('--contrast_factor', type=float, default=0.1, 
                        help="Maximum contrast adjustment factor")
    parser.add_argument('--noise_prob', type=float, default=0.2, 
                        help="Probability of adding noise")
    parser.add_argument('--noise_std', type=float, default=0.01, 
                        help="Standard deviation of noise")
    
    # Patch extraction parameters
    parser.add_argument('--use_patches', action='store_true', 
                        help="Train on patches rather than full images")
    parser.add_argument('--patch_size', type=int, default=64, 
                        help="Size of patches to extract")
    parser.add_argument('--patch_stride', type=int, default=32, 
                        help="Stride between patches")
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16, 
                        help="Batch size")
    parser.add_argument('--epochs', type=int, default=50, 
                        help="Maximum number of training epochs")
    parser.add_argument('--learning_rate', type=float, default=1e-3, 
                        help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=0.0, 
                        help="Weight decay (L2 penalty)")
    parser.add_argument('--momentum', type=float, default=0.9, 
                        help="Momentum for SGD optimizer")
    parser.add_argument('--grad_clip', type=float, default=1.0, 
                        help="Gradient clipping norm (0 to disable)")
    parser.add_argument('--mixed_precision', action='store_true', 
                        help="Enable mixed precision training")
    parser.add_argument('--num_workers', type=int, default=4, 
                        help="Number of worker threads for data loading")
    
    # Optimizer and scheduler parameters
    parser.add_argument('--optimizer_type', type=str, choices=['adam', 'adamw', 'sgd'], default='adam', 
                        help="Type of optimizer to use")
    parser.add_argument('--scheduler_type', type=str, 
                        choices=['plateau', 'cosine', 'onecycle', 'none'], default='plateau', 
                        help="Type of learning rate scheduler to use")
    parser.add_argument('--lr_factor', type=float, default=0.5, 
                        help="Factor by which to reduce learning rate (for ReduceLROnPlateau)")
    parser.add_argument('--lr_patience', type=int, default=5, 
                        help="Patience for learning rate reduction (for ReduceLROnPlateau)")
    parser.add_argument('--min_lr', type=float, default=1e-6, 
                        help="Minimum learning rate")
    parser.add_argument('--warmup_pct', type=float, default=0.3, 
                        help="Percentage of training for learning rate warmup (for OneCycleLR)")
    
    # Loss function parameters
    parser.add_argument('--loss_type', type=str, 
                        choices=['combined', 'l1', 'mse', 'ssim', 'content', 'adaptive'], 
                        default='combined', help="Type of loss function to use")
    parser.add_argument('--ssim_weight', type=float, default=0.5, 
                        help="Weight for SSIM loss (0-1)")
    parser.add_argument('--window_size', type=int, default=11, 
                        help="Window size for SSIM calculation")
    parser.add_argument('--sigma', type=float, default=1.5, 
                        help="Sigma for SSIM calculation")
    parser.add_argument('--use_ms_ssim', action='store_true', 
                        help="Use multi-scale SSIM instead of regular SSIM")
    parser.add_argument('--use_edge_loss', action='store_true', 
                        help="Add edge preservation loss")
    parser.add_argument('--use_freq_loss', action='store_true', 
                        help="Add frequency domain loss")
    parser.add_argument('--edge_weight', type=float, default=0.1, 
                        help="Weight for edge preservation loss")
    parser.add_argument('--freq_weight', type=float, default=0.1, 
                        help="Weight for frequency domain loss")
    
    # Model-specific parameters
    # Simple CNN parameters
    parser.add_argument('--num_blocks', type=int, default=8, 
                        help="Number of residual blocks for simple CNN model")
    parser.add_argument('--num_features', type=int, default=64,
                        help="Number of features for CNN and EDSR models")
    
    # EDSR parameters
    parser.add_argument('--num_res_blocks', type=int, default=16,
                        help="Number of residual blocks for EDSR model")
    parser.add_argument('--res_scale', type=float, default=0.1,
                        help="Residual scaling factor for EDSR model")
    parser.add_argument('--use_mean_shift', action='store_true',
                        help="Use mean shift in EDSR model")
    
    # U-Net parameters
    parser.add_argument('--base_filters', type=int, default=64,
                        help="Number of base filters for U-Net model")
    parser.add_argument('--depth', type=int, default=4,
                        help="Depth of U-Net model")
    parser.add_argument('--norm_type', type=str, choices=['batch', 'instance', 'none'],
                        default='batch', help="Normalization type for U-Net model")
    parser.add_argument('--use_attention', action='store_true',
                        help="Use attention mechanism in U-Net model")
    parser.add_argument('--residual_mode', type=str, choices=['add', 'concat'],
                        default='add', help="Residual connection mode for U-Net model")
    parser.add_argument('--bilinear', action='store_true',
                        help="Use bilinear upsampling in U-Net model")
    
    # Common model parameters
    parser.add_argument('--scale', type=int, default=1,
                        help="Scale factor for super-resolution")
    parser.add_argument('--model_type', type=str, choices=['simple', 'edsr', 'unet'],
                        default='unet', help="Type of model to use")
    
    # Checkpoint and logging parameters
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help="Directory to save checkpoints")
    parser.add_argument('--checkpoint_interval', type=int, default=5,
                        help="Save checkpoint every N epochs")
    parser.add_argument('--resume', action='store_true',
                        help="Resume training from checkpoint")
    parser.add_argument('--use_tensorboard', action='store_true',
                        help="Use TensorBoard for logging")
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help="Directory for TensorBoard logs")
    
    args = parser.parse_args()
    train(args)
