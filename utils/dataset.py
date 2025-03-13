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

class MRISuperResDataset(Dataset):
    """
    Enhanced dataset for MRI super-resolution with:
    - Improved augmentation pipeline
    - Support for different input/output sizes
    - Consistent normalization
    - Subject-aware splitting
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
        return len(self.valid_pairs)
    
    def __getitem__(self, idx):
        # Check if item is in cache
        if idx in self.cache:
            return self.cache[idx]
        
        filename = self.valid_pairs[idx]
        full_res_path = os.path.join(self.full_res_dir, filename)
        low_res_path = os.path.join(self.low_res_dir, filename)
        
        # Load images
        try:
            full_image = Image.open(full_res_path).convert("L")
            low_image = Image.open(low_res_path).convert("L")
        except Exception as e:
            print(f"Error loading images for {filename}: {e}")
            # Return a placeholder if image loading fails
            placeholder = torch.zeros((1, 256, 256))
            return placeholder, placeholder
        
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
            img1_np = np.array(img1).astype(np.float32)
            noise = np.random.normal(0, self.aug_params['noise_std'] * 255, img1_np.shape)
            img1_np = np.clip(img1_np + noise, 0, 255).astype(np.uint8)
            img1 = Image.fromarray(img1_np)
        
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

def create_subject_aware_split(dataset, val_ratio=0.2, test_ratio=0.1, seed=42):
    """
    Create train/validation/test splits that keep all slices from the same subject together.
    
    Args:
        dataset: MRISuperResDataset instance
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data
        seed: Random seed for reproducibility
        
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    random.seed(seed)
    
    # Get unique subjects
    subjects = dataset.get_unique_subjects()
    random.shuffle(subjects)
    
    # Calculate split points
    n_subjects = len(subjects)
    n_val = max(1, int(n_subjects * val_ratio))
    n_test = max(1, int(n_subjects * test_ratio))
    n_train = n_subjects - n_val - n_test
    
    # Split subjects
    train_subjects = subjects[:n_train]
    val_subjects = subjects[n_train:n_train+n_val]
    test_subjects = subjects[n_train+n_val:]
    
    # Get indices for each split
    train_indices = []
    val_indices = []
    test_indices = []
    
    for i, subject in enumerate(dataset.subjects):
        if subject in train_subjects:
            train_indices.append(i)
        elif subject in val_subjects:
            val_indices.append(i)
        elif subject in test_subjects:
            test_indices.append(i)
    
    # Create subset datasets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    print(f"Split dataset into {len(train_dataset)} training, {len(val_dataset)} validation, "
          f"and {len(test_dataset)} test samples.")
    print(f"Using {len(train_subjects)} subjects for training, {len(val_subjects)} for validation, "
          f"and {len(test_subjects)} for testing.")
    
    return train_dataset, val_dataset, test_dataset

class PatchDataset(Dataset):
    """
    Dataset that extracts patches from MRI images for training.
    This can be useful for training on smaller patches rather than full slices.
    """
    def __init__(self, base_dataset, patch_size=64, stride=32, transform=None):
        """
        Args:
            base_dataset: Base dataset to extract patches from
            patch_size: Size of patches to extract
            stride: Stride between patches
            transform: Additional transforms to apply to patches
        """
        self.base_dataset = base_dataset
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        
        # Pre-compute patch indices
        self.patches = []
        for idx in range(len(base_dataset)):
            low_res, full_res = base_dataset[idx]
            h, w = low_res.shape[1], low_res.shape[2]
            
            # Calculate number of patches
            n_h = max(1, (h - patch_size) // stride + 1)
            n_w = max(1, (w - patch_size) // stride + 1)
            
            for i in range(n_h):
                for j in range(n_w):
                    y = min(i * stride, h - patch_size)
                    x = min(j * stride, w - patch_size)
                    self.patches.append((idx, y, x))
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        base_idx, y, x = self.patches[idx]
        low_res, full_res = self.base_dataset[base_idx]
        
        # Extract patch
        low_patch = low_res[:, y:y+self.patch_size, x:x+self.patch_size]
        full_patch = full_res[:, y:y+self.patch_size, x:x+self.patch_size]
        
        # Apply additional transform if provided
        if self.transform:
            low_patch = self.transform(low_patch)
            full_patch = self.transform(full_patch)
        
        return low_patch, full_patch
