import os
import random
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, Subset
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import torch
import re
import logging

class MRISuperResDataset(Dataset):
    """
    Enhanced dataset for MRI super-resolution with:
    - Improved augmentation pipeline
    - Support for different input/output sizes
    - Consistent normalization
    - Metadata tracking
    """
    def __init__(self, full_res_dir, low_res_dir, transform=None, augmentation=True,
                 augmentation_params=None, normalize=True, cache_size=100):
        """
        Args:
            full_res_dir (str): Directory containing full-resolution PNG images.
            low_res_dir (str): Directory containing low-resolution PNG images.
            transform (callable, optional): Transform to apply to images. Defaults to ToTensor.
            augmentation (bool, optional): Whether to apply data augmentation.
            augmentation_params (dict, optional): Parameters for augmentation.
            normalize (bool, optional): Whether to normalize images to [0,1] range.
            cache_size (int, optional): Number of image pairs to cache in memory.
        """
        self.full_res_dir = Path(full_res_dir)
        self.low_res_dir = Path(low_res_dir)
        
        # Find all PNG files in full-res directory
        self.full_res_files = sorted([
            f for f in os.listdir(full_res_dir) if f.lower().endswith(".png")
        ])
        
        # Verify that corresponding low-res files exist
        self.valid_pairs = []
        self.subjects = []
        self.metadata = []
        
        for f in self.full_res_files:
            low_res_path = self.low_res_dir / f
            if low_res_path.exists():
                self.valid_pairs.append(f)
                
                # Extract subject ID from filename using regex
                # Assuming format like "sub-HC001_ses-01_acq-inv1_T1map.nii_s064.png"
                match = re.search(r'sub-([A-Za-z0-9]+)', f)
                if match:
                    subject_id = match.group(1)
                    self.subjects.append(subject_id)
                else:
                    # If no subject ID found, use filename as subject
                    self.subjects.append(f)
                
                # Store metadata
                self.metadata.append({
                    'filename': f,
                    'subject': self.subjects[-1],
                    'full_res_path': str(self.full_res_dir / f),
                    'low_res_path': str(self.low_res_dir / f)
                })
        
        # Set up augmentation parameters
        self.augmentation = augmentation
        self.default_aug_params = {
            'flip_prob': 0.5,
            'rotate_prob': 0.5,
            'rotate_range': (-5, 5),
            'brightness_prob': 0.3,
            'brightness_range': (0.9, 1.1),
            'contrast_prob': 0.3,
            'contrast_range': (0.9, 1.1),
            'noise_prob': 0.2,
            'noise_std': 0.01
        }
        
        if augmentation_params is not None:
            self.aug_params = {**self.default_aug_params, **augmentation_params}
        else:
            self.aug_params = self.default_aug_params
        
        # Set up transform
        self.transform = transform if transform is not None else transforms.ToTensor()
        self.normalize = normalize
        
        # Set up cache
        self.cache_size = cache_size
        self.cache = {}
    
    def __len__(self):
        return len(self.full_res_files)
    
    def __getitem__(self, idx):
        # Check if item is in cache
        if idx in self.cache:
            return self.cache[idx]
        
        filename = self.full_res_files[idx]
        full_res_path = os.path.join(self.full_res_dir, filename)
        low_res_path = os.path.join(self.low_res_dir, filename)
        
        # Load images
        try:
            full_image = Image.open(full_res_path).convert("L")
            low_image = Image.open(low_res_path).convert("L")
        except Exception as e:
            error_msg = f"Error loading images for {filename} at paths: {full_res_path} and {low_res_path}. Error: {e}"
            print(error_msg)
            logging.error(error_msg)
            # Raise exception instead of returning placeholder to avoid silent failures
            raise RuntimeError(error_msg)
        
        # Apply augmentation if enabled
        if self.augmentation:
            full_image, low_image = self.augment_pair(full_image, low_image)
        
        # Apply transform
        full_tensor = self.transform(full_image)
        low_tensor = self.transform(low_image)
        
        # Apply normalization if enabled
        if self.normalize:
            # Ensure values are in [0, 1]
            if full_tensor.max() > 1.0:
                full_tensor = full_tensor / 255.0
            if low_tensor.max() > 1.0:
                low_tensor = low_tensor / 255.0
        
        # Cache result if cache is enabled
        if self.cache_size > 0:
            # Remove oldest item if cache is full
            if len(self.cache) >= self.cache_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            
            self.cache[idx] = (low_tensor, full_tensor)
        
        return low_tensor, full_tensor
    
    def augment_pair(self, img1, img2):
        """
        Apply identical augmentation to both images with enhanced options.
        """
        # Horizontal flip
        if random.random() < self.aug_params['flip_prob']:
            img1 = TF.hflip(img1)
            img2 = TF.hflip(img2)
        
        # Random rotation
        if random.random() < self.aug_params['rotate_prob']:
            angle = random.uniform(*self.aug_params['rotate_range'])
            # Use the image's mean intensity as fill value
            fill_val1 = int(TF.to_tensor(img1).mean().item() * 255)
            fill_val2 = int(TF.to_tensor(img2).mean().item() * 255)
            img1 = TF.rotate(img1, angle, fill=fill_val1)
            img2 = TF.rotate(img2, angle, fill=fill_val2)
        
        # Brightness adjustment
        if random.random() < self.aug_params['brightness_prob']:
            brightness_factor = random.uniform(*self.aug_params['brightness_range'])
            img1 = TF.adjust_brightness(img1, brightness_factor)
            img2 = TF.adjust_brightness(img2, brightness_factor)
        
        # Contrast adjustment
        if random.random() < self.aug_params['contrast_prob']:
            contrast_factor = random.uniform(*self.aug_params['contrast_range'])
            img1 = TF.adjust_contrast(img1, contrast_factor)
            img2 = TF.adjust_contrast(img2, contrast_factor)
        
        # Add random noise (only to low-res image to simulate scanner noise)
        if random.random() < self.aug_params['noise_prob']:
            img2_np = np.array(img2).astype(np.float32)  # img2 is the low-res image
            noise = np.random.normal(0, self.aug_params['noise_std'] * 255, img2_np.shape)
            img2_np = np.clip(img2_np + noise, 0, 255).astype(np.uint8)
            img2 = Image.fromarray(img2_np)
        
        return img1, img2
    
    def get_subject_indices(self, subject_id):
        """
        Get all indices belonging to a specific subject.
        """
        return [i for i, s in enumerate(self.subjects) if s == subject_id]
    
    def get_unique_subjects(self):
        """
        Get list of unique subject IDs in the dataset.
        """
        return list(set(self.subjects))
