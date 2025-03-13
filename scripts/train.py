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
from utils.dataset import MRISuperResDataset, create_subject_aware_split, PatchDataset, MRIDataset
from utils.losses import CombinedLoss, PSNR, MS_SSIM, ContentLoss, AdaptiveLoss, ssim, SSIM
from models.cnn_model import CNNSuperRes
from models.edsr_model import EDSRSuperRes
from models.unet_model import UNetSuperRes
from torchvision import transforms
from torch.utils.data import random_split
from contextlib import nullcontext

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
    """Train the MRI quality enhancement model"""
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Create output directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    if args.use_tensorboard:
        os.makedirs(args.log_dir, exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize gradient scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision)

    # Create model
    if args.model_type == 'unet':
        model = UNetSuperRes(
            in_channels=1,
            out_channels=1,
            base_filters=args.base_filters
        )
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    # Move model to device
    model = model.to(device)

    # Create optimizer
    if args.optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer_type == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer type: {args.optimizer_type}")

    # Create scheduler
    if args.scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler_type == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_factor, patience=args.lr_patience)
    else:
        raise ValueError(f"Unsupported scheduler type: {args.scheduler_type}")

    # Create loss functions
    criterion = CombinedLoss(
        alpha=args.ssim_weight,
        window_size=args.window_size,
        sigma=args.sigma,
        val_range=1.0,
        device=device,
        use_ms_ssim=args.use_ms_ssim,
        use_edge_loss=args.use_edge_loss,
        edge_weight=args.edge_weight,
        use_freq_loss=args.use_freq_loss,
        freq_weight=args.freq_weight
    )

    # Create datasets
    train_dataset = MRISuperResDataset(
        full_res_dir=args.full_res_dir,
        low_res_dir=args.low_res_dir,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ]),
        augmentation=args.augmentation,
        augmentation_params={
            'flip_prob': args.flip_prob,
            'rotate_prob': args.rotate_prob,
            'rotate_range': (-args.rotate_angle, args.rotate_angle),
            'brightness_prob': args.brightness_prob,
            'brightness_range': (1.0-args.brightness_factor, 1.0+args.brightness_factor),
            'contrast_prob': args.contrast_prob,
            'contrast_range': (1.0-args.contrast_factor, 1.0+args.contrast_factor),
            'noise_prob': args.noise_prob,
            'noise_std': args.noise_std
        } if args.augmentation else None
    )

    # Split dataset
    train_size = int((1 - args.validation_split - args.test_split) * len(train_dataset))
    val_size = int(args.validation_split * len(train_dataset))
    test_size = len(train_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(train_dataset, [train_size, val_size, test_size])

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Create metrics
    metrics = {
        'psnr': PSNR(),
        'ssim': SSIM(window_size=args.window_size, sigma=args.sigma),
        'ms_ssim': MS_SSIM(window_size=args.window_size, sigma=args.sigma) if args.use_ms_ssim else None
    }

    # Create tensorboard writer
    writer = SummaryWriter(args.log_dir) if args.use_tensorboard else None

    # Training loop
    best_val_loss = float('inf')
    best_model_path = None
    patience_counter = 0

    # Log training parameters
    print("\nTraining Parameters:")
    print(f"Model Type: {args.model_type}")
    print(f"Base Filters: {args.base_filters}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Optimizer: {args.optimizer_type}")
    print(f"Scheduler: {args.scheduler_type}")
    print(f"Loss Type: {args.loss_type}")
    print(f"SSIM Weight: {args.ssim_weight}")
    print(f"Edge Loss: {args.use_edge_loss}")
    print(f"Frequency Loss: {args.use_freq_loss}")
    print(f"Data Augmentation: {args.augmentation}")
    print(f"Mixed Precision: {args.mixed_precision}")
    print(f"Patch Size: {args.patch_size if args.use_patches else 'None'}")
    print(f"Patch Stride: {args.patch_stride if args.use_patches else 'None'}")
    print(f"Validation Split: {args.validation_split}")
    print(f"Test Split: {args.test_split}")
    print(f"Learning Rate Patience: {args.lr_patience}")
    print(f"Warmup Percentage: {args.warmup_pct}")
    print(f"Gradient Clipping: {args.grad_clip}")
    print(f"Weight Decay: {args.weight_decay}")
    print(f"Momentum: {args.momentum}")
    print(f"Number of Workers: {args.num_workers}")
    print(f"Seed: {args.seed}")
    print("\nStarting training...\n")

    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_metrics = {name: 0 for name in metrics.keys() if metrics[name] is not None}
        train_batches = 0

        for batch_idx, (low_res, high_res) in enumerate(train_loader):
            low_res = low_res.to(device)
            high_res = high_res.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast() if args.mixed_precision else nullcontext():
                output = model(low_res)
                loss = criterion(output, high_res)

            if args.mixed_precision:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            train_loss += loss.item()
            for name, metric in metrics.items():
                if metric is not None:
                    train_metrics[name] += metric(output, high_res).item()
            train_batches += 1

            if batch_idx % args.log_interval == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(low_res)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

        # Calculate average training metrics
        train_loss /= train_batches
        for name in train_metrics:
            train_metrics[name] /= train_batches

        # Validation phase
        model.eval()
        val_loss = 0
        val_metrics = {name: 0 for name in metrics.keys() if metrics[name] is not None}
        val_batches = 0

        with torch.no_grad():
            for low_res, high_res in val_loader:
                low_res = low_res.to(device)
                high_res = high_res.to(device)

                with torch.cuda.amp.autocast() if args.mixed_precision else nullcontext():
                    output = model(low_res)
                    loss = criterion(output, high_res)

                val_loss += loss.item()
                for name, metric in metrics.items():
                    if metric is not None:
                        val_metrics[name] += metric(output, high_res).item()
                val_batches += 1

        # Calculate average validation metrics
        val_loss /= val_batches
        for name in val_metrics:
            val_metrics[name] /= val_batches

        # Update learning rate
        if args.scheduler_type == 'reduce_on_plateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()

        # Log metrics
        print(f'\nEpoch {epoch}:')
        print(f'Training Loss: {train_loss:.6f}')
        print(f'Validation Loss: {val_loss:.6f}')
        for name in train_metrics:
            print(f'Training {name.upper()}: {train_metrics[name]:.6f}')
            print(f'Validation {name.upper()}: {val_metrics[name]:.6f}')

        if writer is not None:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            for name in train_metrics:
                writer.add_scalar(f'{name.upper()}/train', train_metrics[name], epoch)
                writer.add_scalar(f'{name.upper()}/val', val_metrics[name], epoch)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            if best_model_path is not None:
                os.remove(best_model_path)
            best_model_path = os.path.join(args.checkpoint_dir, f'best_model_{args.model_type}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'args': args
            }, best_model_path)
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= args.lr_patience:
            print(f'\nEarly stopping triggered after {epoch + 1} epochs')
            break

    # Test phase
    print("\nTesting best model...")
    model.load_state_dict(torch.load(best_model_path)['model_state_dict'])
    model.eval()
    test_loss = 0
    test_metrics = {name: 0 for name in metrics.keys() if metrics[name] is not None}
    test_batches = 0

    with torch.no_grad():
        for low_res, high_res in test_loader:
            low_res = low_res.to(device)
            high_res = high_res.to(device)

            with torch.cuda.amp.autocast() if args.mixed_precision else nullcontext():
                output = model(low_res)
                loss = criterion(output, high_res)

            test_loss += loss.item()
            for name, metric in metrics.items():
                if metric is not None:
                    test_metrics[name] += metric(output, high_res).item()
            test_batches += 1

    # Calculate average test metrics
    test_loss /= test_batches
    for name in test_metrics:
        test_metrics[name] /= test_batches

    # Log final test results
    print("\nFinal Test Results:")
    print(f'Test Loss: {test_loss:.6f}')
    for name in test_metrics:
        print(f'Test {name.upper()}: {test_metrics[name]:.6f}')

    if writer is not None:
        writer.add_scalar('Loss/test', test_loss, args.epochs)
        for name in test_metrics:
            writer.add_scalar(f'{name.upper()}/test', test_metrics[name], args.epochs)
        writer.close()

    return best_model_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train MRI quality enhancement model")
    
    # Data paths
    parser.add_argument('--full_res_dir', type=str, required=True,
                      help='Directory containing high-quality MRI slices')
    parser.add_argument('--low_res_dir', type=str, required=True,
                      help='Directory containing low-quality MRI slices')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='unet',
                      choices=['unet'], help='Model architecture')
    parser.add_argument('--base_filters', type=int, default=32,
                      help='Number of base filters in the model')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8,
                      help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=200,
                      help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                      help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                      help='Weight decay for optimizer')
    parser.add_argument('--momentum', type=float, default=0.9,
                      help='Momentum for SGD optimizer')
    
    # Optimization
    parser.add_argument('--optimizer_type', type=str, default='adamw',
                      choices=['adam', 'adamw', 'sgd'], help='Optimizer type')
    parser.add_argument('--scheduler_type', type=str, default='cosine',
                      choices=['cosine', 'reduce_on_plateau'], help='Learning rate scheduler type')
    parser.add_argument('--lr_factor', type=float, default=0.5,
                      help='Factor to reduce learning rate by')
    parser.add_argument('--lr_patience', type=int, default=5,
                      help='Number of epochs to wait before reducing learning rate')
    parser.add_argument('--warmup_pct', type=float, default=0.3,
                      help='Percentage of training to use for warmup')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                      help='Gradient clipping value')
    
    # Loss function
    parser.add_argument('--loss_type', type=str, default='combined',
                      choices=['l1', 'l2', 'combined'], help='Loss function type')
    parser.add_argument('--ssim_weight', type=float, default=0.7,
                      help='Weight for SSIM loss component')
    parser.add_argument('--window_size', type=int, default=11,
                      help='Window size for SSIM calculation')
    parser.add_argument('--sigma', type=float, default=1.5,
                      help='Sigma for SSIM calculation')
    parser.add_argument('--use_ms_ssim', action='store_true',
                      help='Use MS-SSIM instead of SSIM')
    parser.add_argument('--use_edge_loss', action='store_true',
                      help='Use edge loss component')
    parser.add_argument('--edge_weight', type=float, default=0.2,
                      help='Weight for edge loss component')
    parser.add_argument('--use_freq_loss', action='store_true',
                      help='Use frequency domain loss component')
    parser.add_argument('--freq_weight', type=float, default=0.1,
                      help='Weight for frequency loss component')
    
    # Data augmentation
    parser.add_argument('--augmentation', action='store_true',
                      help='Enable data augmentation')
    parser.add_argument('--flip_prob', type=float, default=0.5,
                      help='Probability of horizontal/vertical flips')
    parser.add_argument('--rotate_prob', type=float, default=0.5,
                      help='Probability of rotation')
    parser.add_argument('--rotate_angle', type=float, default=5.0,
                      help='Maximum rotation angle in degrees')
    parser.add_argument('--brightness_prob', type=float, default=0.3,
                      help='Probability of brightness adjustment')
    parser.add_argument('--brightness_factor', type=float, default=0.1,
                      help='Maximum brightness adjustment factor')
    parser.add_argument('--contrast_prob', type=float, default=0.3,
                      help='Probability of contrast adjustment')
    parser.add_argument('--contrast_factor', type=float, default=0.1,
                      help='Maximum contrast adjustment factor')
    parser.add_argument('--noise_prob', type=float, default=0.2,
                      help='Probability of adding noise')
    parser.add_argument('--noise_std', type=float, default=0.01,
                      help='Standard deviation of noise')
    
    # Training settings
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                      help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs',
                      help='Directory to save logs')
    parser.add_argument('--validation_split', type=float, default=0.2,
                      help='Fraction of data to use for validation')
    parser.add_argument('--test_split', type=float, default=0.1,
                      help='Fraction of data to use for testing')
    parser.add_argument('--num_workers', type=int, default=2,
                      help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    parser.add_argument('--mixed_precision', action='store_true',
                      help='Use mixed precision training')
    parser.add_argument('--use_tensorboard', action='store_true',
                      help='Use TensorBoard for logging')
    parser.add_argument('--log_interval', type=int, default=10,
                      help='Number of batches between logging')
    
    args = parser.parse_args()
    train(args)
