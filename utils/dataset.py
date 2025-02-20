import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import random

class MRISuperResDataset(Dataset):
    def __init__(self, full_res_dir, low_res_dir, transform=None, augmentation=True):
        """
        Expects that the full-resolution and downsampled images share the same filenames.
        
        Args:
            full_res_dir (str): Directory containing full-resolution PNG images.
            low_res_dir (str): Directory containing low-resolution PNG images.
            transform (callable, optional): Transform to apply to images. Defaults to transforms.ToTensor().
            augmentation (bool, optional): Whether to apply random orientation augmentations.
        """
        self.full_res_dir = full_res_dir
        self.low_res_dir = low_res_dir
        self.full_res_files = sorted([f for f in os.listdir(full_res_dir) if f.lower().endswith('.png')])
        self.augmentation = augmentation
        self.transform = transform if transform is not None else transforms.ToTensor()

    def __len__(self):
        return len(self.full_res_files)

    def __getitem__(self, idx):
        full_res_filename = self.full_res_files[idx]
        low_res_filename = full_res_filename  # Filenames must match across directories.
        full_res_path = os.path.join(self.full_res_dir, full_res_filename)
        low_res_path = os.path.join(self.low_res_dir, low_res_filename)
        
        full_image = Image.open(full_res_path).convert('L')
        low_image = Image.open(low_res_path).convert('L')
        
        # Apply on-the-fly paired augmentation if enabled.
        if self.augmentation:
            full_image, low_image = self.augment_pair(full_image, low_image)
        
        # Apply transform (e.g., convert to tensor).
        if self.transform:
            full_image = self.transform(full_image)
            low_image = self.transform(low_image)
        
        return low_image, full_image

    def augment_pair(self, img1, img2):
        """Apply identical random augmentation to both images."""
        # Random horizontal flip.
        if random.random() > 0.5:
            img1 = TF.hflip(img1)
            img2 = TF.hflip(img2)
        # Random rotation between -10 and 10 degrees.
        angle = random.uniform(-10, 10)
        img1 = TF.rotate(img1, angle)
        img2 = TF.rotate(img2, angle)
        return img1, img2
