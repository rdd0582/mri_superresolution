import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
import torchvision.models as models
import torchvision.transforms as transforms

# VGG normalization constants
VGG_MEAN = [0.485, 0.456, 0.406]
VGG_STD = [0.229, 0.224, 0.225]

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

class VGGFeatureExtractor(nn.Module):
    """
    Helper module to extract features from intermediate layers of a VGG network.
    """
    def __init__(self, feature_layer_idx=35, use_maxpool=False):
        super().__init__()
        # Load VGG19 (or VGG16) - VGG19 is often preferred for perceptual loss
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        
        # Choose the features up to the specified layer
        # Layer 35 corresponds to relu5_4 in VGG19, a common choice.
        # Adjust the index based on the desired feature level.
        self.features = nn.Sequential(*list(vgg.children())[:(feature_layer_idx + 1)])
        
        # Freeze VGG parameters - we don't want to train it
        for param in self.features.parameters():
            param.requires_grad = False
            
        # Register normalization buffer - needs to be on the same device as input
        self.register_buffer('mean', torch.tensor(VGG_MEAN).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(VGG_STD).view(1, 3, 1, 1))

    def forward(self, x):
        # 1. Ensure input is 3 channels (replicate grayscale if needed)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            
        # 2. Normalize input using VGG's expected mean and std
        # Ensure normalization tensors have the same dtype and device as input
        mean = self.mean.to(dtype=x.dtype, device=x.device)
        std = self.std.to(dtype=x.dtype, device=x.device)
        x = (x - mean) / std
        
        # 3. Extract features
        output = self.features(x)
        return output

class PerceptualLoss(nn.Module):
    """
    Calculates the perceptual loss between two images using VGG features.
    Loss is typically the L1 or L2 distance between feature maps.
    """
    def __init__(self, feature_layer_idx=35, loss_type='l1'):
        super().__init__()
        self.feature_extractor = VGGFeatureExtractor(feature_layer_idx=feature_layer_idx)
        if loss_type == 'l1':
            self.criterion = nn.L1Loss()
        elif loss_type == 'l2' or loss_type == 'mse':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss type for PerceptualLoss: {loss_type}")
            
        # Ensure feature extractor is in eval mode
        self.feature_extractor.eval()

    def forward(self, generated, target):
        # Ensure feature extractor is on the correct device
        if next(self.feature_extractor.parameters()).device != generated.device:
            self.feature_extractor.to(generated.device)
            
        # Extract features for both images
        # Detach target features as we don't need gradients for the target image
        features_gen = self.feature_extractor(generated)
        with torch.no_grad():
            features_target = self.feature_extractor(target)
        
        # Calculate the loss between feature maps
        loss = self.criterion(features_gen, features_target)
        return loss

class CombinedLoss(nn.Module):
    """
    Combined loss function: L1, SSIM, and optional Perceptual Loss.
    Total loss = l1_weight * L1 + ssim_weight * (1 - SSIM) + perceptual_weight * Perceptual
    where l1_weight = 1 - ssim_weight - perceptual_weight (implicitly calculated if perceptual is used)
    """
    def __init__(self, ssim_weight=0.5, perceptual_weight=0.0, 
                 vgg_layer_idx=35, # Default VGG layer (relu5_4 for VGG19)
                 perceptual_loss_type='l1',
                 window_size=11, sigma=1.5, val_range=1.0, 
                 device=torch.device("cpu")):
        super().__init__()
        
        if not (0 <= ssim_weight <= 1):
            raise ValueError("ssim_weight must be between 0 and 1")
        if not (0 <= perceptual_weight <= 1):
             raise ValueError("perceptual_weight must be between 0 and 1")
        if ssim_weight + perceptual_weight > 1:
             raise ValueError("Sum of ssim_weight and perceptual_weight cannot exceed 1")
             
        self.ssim_weight = ssim_weight
        self.perceptual_weight = perceptual_weight
        self.l1_weight = 1.0 - ssim_weight - perceptual_weight # Derived weight for L1

        self.l1_loss = nn.L1Loss()
        
        # SSIM components
        self.window_size = window_size
        self.sigma = sigma
        self.val_range = val_range
        self.device = device
        self.register_buffer("window", create_window(window_size, 1, sigma, device)) # Assuming single channel input for SSIM

        # Perceptual loss component (optional)
        self.use_perceptual = perceptual_weight > 0
        if self.use_perceptual:
            self.perceptual_loss = PerceptualLoss(
                feature_layer_idx=vgg_layer_idx, 
                loss_type=perceptual_loss_type
            ).to(device) # Ensure perceptual loss module is on the correct device initially
            # Freeze parameters just in case (already done in VGGFeatureExtractor, but good practice)
            for param in self.perceptual_loss.parameters():
                param.requires_grad = False
            self.perceptual_loss.eval() # Ensure it's in eval mode
        else:
            self.perceptual_loss = None
            
    def forward(self, output, target):
        total_loss = 0.0
        losses = {} # Dictionary to store individual loss components if needed

        # --- L1 Loss ---
        if self.l1_weight > 0:
            l1_loss_val = self.l1_loss(output, target)
            total_loss += self.l1_weight * l1_loss_val
            losses['l1_loss'] = l1_loss_val.item() # Store value

        # --- SSIM Loss ---
        if self.ssim_weight > 0:
            # Ensure window has the same dtype and device as input tensors
            window = self.window
            if window.dtype != output.dtype or window.device != output.device:
                window = window.to(device=output.device, dtype=output.dtype)
                
            ssim_val = ssim(output, target, self.window_size, self.sigma, 
                           self.val_range, output.device, window)
            
            # Clamp SSIM to [0, 1] for stability
            ssim_val = torch.clamp(ssim_val, min=0.0, max=1.0)
            ssim_loss = 1 - ssim_val
            
            total_loss += self.ssim_weight * ssim_loss
            losses['ssim_loss'] = ssim_loss.item() # Store value (1-ssim)
            losses['ssim_metric'] = ssim_val.item() # Also store the direct ssim value

        # --- Perceptual Loss ---
        if self.use_perceptual and self.perceptual_loss is not None:
             # Ensure the perceptual loss module is on the same device as the input
            if next(self.perceptual_loss.parameters()).device != output.device:
                 self.perceptual_loss.to(output.device)
                 
            perceptual_loss_val = self.perceptual_loss(output, target)
            total_loss += self.perceptual_weight * perceptual_loss_val
            losses['perceptual_loss'] = perceptual_loss_val.item() # Store value
        
        # Return total loss (and optionally the dictionary of individual losses for logging)
        # For now, just return the combined scalar loss for backpropagation
        return total_loss

class SSIM(nn.Module):
    """
    Structural Similarity Index (SSIM) as a PyTorch module.
    Wraps the ssim function for more convenient use as a metric.
    """
    def __init__(self, window_size=11, sigma=1.5, val_range=1.0, device=None):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.val_range = val_range
        
        self.device = device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Ensure window is created on the correct initial device
        self.register_buffer("window", create_window(window_size, 1, sigma, self.device)) # Assuming single channel 
    
    def forward(self, img1, img2):
        # Ensure the window buffer is on the same device as the input images for the calculation
        window = self.window
        if window.device != img1.device:
            window = window.to(img1.device)
            
        return ssim(img1, img2, self.window_size, self.sigma, 
                   self.val_range, img1.device, window) # Use img1.device
