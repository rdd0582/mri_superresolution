import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# The SSIM-related functions are kept for compatibility but are not used.
def gaussian_window(window_size: int, sigma: float):
    """Creates a 1D Gaussian window."""
    gauss = torch.tensor(
        [math.exp(-(x - window_size // 2)**2 / (2 * sigma**2)) for x in range(window_size)],
        dtype=torch.float32
    )
    return gauss / gauss.sum()

def create_window(window_size: int, channel: int, sigma: float, device: torch.device):
    """Creates a 2D Gaussian window (kernel) for SSIM computation."""
    _1D_window = gaussian_window(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window @ _1D_window.t()  # Outer product to get 2D window
    window = _2D_window.expand(channel, 1, window_size, window_size).to(device)
    return window

def ssim(img1, img2, window_size=11, sigma=1.5, val_range=1.0, device=torch.device("cpu"), window=None):
    """
    Compute SSIM index between img1 and img2.
    If a precomputed window is provided, it is used directly.
    This function is retained for compatibility but is not used in CombinedLoss.
    """
    channel = img1.size(1)
    if window is None:
        window = create_window(window_size, channel, sigma, device)
    
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    # Constants for numerical stability.
    C1 = (0.01 * val_range) ** 2
    C2 = (0.03 * val_range) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

class CombinedLoss(nn.Module):
    """
    A plug-and-play loss function originally combining L1 and SSIM losses:
      loss = alpha * (1 - SSIM) + (1 - alpha) * L1_loss
    For your purposes, only the L1 loss is used.
    """
    def __init__(self, alpha=0.85, window_size=11, sigma=1.5, val_range=1.0, device=torch.device("cpu")):
        super().__init__()
        self.alpha = alpha  # Retained for compatibility.
        self.l1_loss = nn.L1Loss()
        self.window_size = window_size
        self.sigma = sigma
        self.val_range = val_range
        self.device = device
        # The SSIM window is still precomputed for plug-and-play compatibility.
        self.register_buffer("window", create_window(window_size, 1, sigma, device))
    
    def forward(self, output, target):
        # Instead of combining SSIM and L1, only the L1 loss is computed.
        loss = self.l1_loss(output, target)
        return loss
