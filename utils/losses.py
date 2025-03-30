import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable

def gaussian_window(window_size: int, sigma: float):
    """Creates a 1D Gaussian window."""
    # More efficient implementation using torch.linspace
    coords = torch.arange(window_size, dtype=torch.float32)
    coords -= window_size // 2
    
    # Gaussian function
    gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return gauss / gauss.sum()

def create_window(window_size: int, channel: int, sigma: float, device: torch.device):
    """Creates a 2D Gaussian window (kernel) for SSIM computation."""
    _1D_window = gaussian_window(window_size, sigma).to(device).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, sigma=1.5, val_range=1.0, device=None, window=None, size_average=True):
    """
    Compute SSIM index between img1 and img2.
    Improved implementation with better memory efficiency.
    """
    # For metrics calculation, using float32 is more stable
    # Store original dtypes and device
    orig_dtype1 = img1.dtype
    orig_dtype2 = img2.dtype
    
    if device is None:
        device = img1.device
        
    # Ensure both images are on the same device and have same dtype (use float32 for reliability)
    img1 = img1.to(device=device, dtype=torch.float32)
    img2 = img2.to(device=device, dtype=torch.float32)
        
    channel = img1.size(1)
    
    if window is None:
        window = create_window(window_size, channel, sigma, device)
    
    # Always ensure window is float32 on the right device
    window = window.to(device=device, dtype=torch.float32)
    
    # Pad images if needed
    pad = window_size // 2
    
    mu1 = F.conv2d(img1, window, padding=pad, groups=channel)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channel) - mu1_mu2
    
    # Constants for numerical stability
    C1 = (0.01 * val_range) ** 2
    C2 = (0.03 * val_range) ** 2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        result = ssim_map.mean()
    else:
        result = ssim_map.mean(1).mean(1).mean(1)
        
    # Convert back to original dtype if needed
    if orig_dtype1 != torch.float32 and orig_dtype1 == orig_dtype2:
        result = result.to(dtype=orig_dtype1)
    
    return result

class CombinedLoss(nn.Module):
    """
    Simple combined loss function with L1 and SSIM components:
    - L1 Loss for pixel-wise accuracy
    - SSIM for structural similarity
    
    Total loss = alpha * (1 - SSIM) + (1 - alpha) * L1_loss
    """
    def __init__(self, alpha=0.5, window_size=11, sigma=1.5, val_range=1.0, 
                 device=torch.device("cpu")):
        super().__init__()
        self.alpha = alpha
        self.l1_loss = nn.L1Loss()
        self.window_size = window_size
        self.sigma = sigma
        self.val_range = val_range
        self.device = device
        
        # Register window buffer for SSIM
        self.register_buffer("window", create_window(window_size, 1, sigma, device))
    
    def forward(self, output, target):
        # Compute L1 loss (pixel-wise)
        l1_loss_val = self.l1_loss(output, target)
        
        # Compute structural similarity loss
        # Ensure window has the same dtype as input tensors
        window = self.window
        if window.dtype != output.dtype or window.device != output.device:
            window = window.to(device=output.device, dtype=output.dtype)
            
        ssim_val = ssim(output, target, self.window_size, self.sigma, 
                       self.val_range, output.device, window)
        
        # Clamp the SSIM value to the range [0, 1]
        # This prevents ssim_loss from becoming negative if ssim_val slightly exceeds 1
        # Clamping to min=0 also handles potential rare cases where SSIM might be negative
        ssim_val = torch.clamp(ssim_val, min=0.0, max=1.0)
        
        ssim_loss = 1 - ssim_val
        
        # Combined loss
        combined_loss = self.alpha * ssim_loss + (1 - self.alpha) * l1_loss_val
        
        return combined_loss

class SSIM(nn.Module):
    """
    Structural Similarity Index (SSIM) as a PyTorch module.
    Wraps the ssim function for more convenient use.
    """
    def __init__(self, window_size=11, sigma=1.5, val_range=1.0, device=None):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.val_range = val_range
        
        self.device = device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.register_buffer("window", create_window(window_size, 1, sigma, self.device))
    
    def forward(self, img1, img2):
        return ssim(img1, img2, self.window_size, self.sigma, 
                   self.val_range, self.device, self.window)
