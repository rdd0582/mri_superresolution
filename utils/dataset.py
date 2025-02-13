import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class MRISuperResDataset(Dataset):
    def __init__(self, full_res_dir, low_res_dir, transform=None):
        """
        Expects that the full-resolution and downsampled images share the same filenames.
        """
        self.full_res_dir = full_res_dir
        self.low_res_dir = low_res_dir
        self.full_res_files = sorted([f for f in os.listdir(full_res_dir) if f.lower().endswith('.png')])
        self.transform = transform if transform is not None else transforms.ToTensor()

    def __len__(self):
        return len(self.full_res_files)

    def __getitem__(self, idx):
        full_res_filename = self.full_res_files[idx]
        low_res_filename = full_res_filename  # filenames must match across directories
        full_res_path = os.path.join(self.full_res_dir, full_res_filename)
        low_res_path = os.path.join(self.low_res_dir, low_res_filename)
        full_image = Image.open(full_res_path).convert('L')
        low_image = Image.open(low_res_path).convert('L')
        if self.transform:
            full_image = self.transform(full_image)
            low_image = self.transform(low_image)
        return low_image, full_image
