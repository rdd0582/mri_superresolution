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

class MS_SSIM(nn.Module):
    """
    Multi-Scale Structural Similarity Index (MS-SSIM)
    Evaluates image similarity at multiple scales for more robust assessment
    """
    def __init__(self, window_size=11, sigma=1.5, val_range=1.0, device=torch.device("cpu"), 
                 weights=None, levels=5):
        super(MS_SSIM, self).__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.val_range = val_range
        self.device = device
        
        # Default weights for MS-SSIM
        if weights is None:
            weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
        
        self.weights = weights
        self.levels = levels
        self.register_buffer("window", create_window(window_size, 1, sigma, device))
    
    def forward(self, img1, img2):
        # Convert inputs to float32 for stable computation
        orig_dtype = img1.dtype
        device = img1.device
        img1 = img1.to(dtype=torch.float32)
        img2 = img2.to(dtype=torch.float32)
        
        # Use float32 weights and window
        weights = self.weights.to(device=device, dtype=torch.float32)
        window = self.window.to(device=device, dtype=torch.float32)
        
        levels = self.levels
        
        # Check if images are too small for the requested number of levels
        min_size = min(img1.size(2), img1.size(3))
        max_levels = int(math.log2(min_size)) - 2  # Ensure minimum size of 9 at coarsest level
        levels = min(levels, max_levels)
        
        # Adjust weights if levels changed
        if levels < len(weights):
            weights = weights[:levels] / weights[:levels].sum()
        
        mssim = []
        mcs = []
        
        for i in range(levels):
            ssim_val, cs = self._ssim(img1, img2, window, self.val_range, return_cs=True)
            mssim.append(ssim_val)
            mcs.append(cs)
            
            if i < levels - 1:
                img1 = F.avg_pool2d(img1, kernel_size=2, stride=2)
                img2 = F.avg_pool2d(img2, kernel_size=2, stride=2)
        
        # Convert lists to tensors
        mssim = torch.stack(mssim)
        mcs = torch.stack(mcs)
        
        # Calculate MS-SSIM
        # Use the last scale SSIM and the product of all contrast sensitivities
        msssim = torch.prod(mcs[:-1] ** weights[:-1]) * (mssim[-1] ** weights[-1])
        
        # Convert back to original dtype if needed
        if orig_dtype != torch.float32:
            msssim = msssim.to(dtype=orig_dtype)
        
        return msssim
    
    def _ssim(self, img1, img2, window, val_range, size_average=True, return_cs=False):
        """Internal SSIM calculation with contrast sensitivity option"""
        # Ensure all inputs are float32
        if img1.dtype != torch.float32:
            img1 = img1.to(dtype=torch.float32)
        if img2.dtype != torch.float32:
            img2 = img2.to(dtype=torch.float32)
        if window.dtype != torch.float32:
            window = window.to(dtype=torch.float32)
            
        channel = img1.size(1)
        
        # Pad images if needed
        pad = self.window_size // 2
        
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
        
        # Contrast sensitivity
        cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
        # SSIM
        ssim_map = ((2 * mu1_mu2 + C1) * cs_map) / ((mu1_sq + mu2_sq + C1))
        
        if size_average:
            cs = cs_map.mean()
            ssim_val = ssim_map.mean()
        else:
            cs = cs_map.mean(1).mean(1).mean(1)
            ssim_val = ssim_map.mean(1).mean(1).mean(1)
            
        if return_cs:
            return ssim_val, cs
        else:
            return ssim_val

class CombinedLoss(nn.Module):
    """
    Enhanced combined loss function with multiple components:
    - L1 Loss for pixel-wise accuracy
    - SSIM or MS-SSIM for structural similarity
    - Optional edge detection loss for preserving boundaries
    - Optional frequency domain loss
    
    Allows dynamic weighting of components during training.
    """
    def __init__(self, alpha=0.5, window_size=11, sigma=1.5, val_range=1.0, 
                 device=torch.device("cpu"), use_ms_ssim=False, use_edge_loss=False,
                 use_freq_loss=False, edge_weight=0.1, freq_weight=0.1):
        super().__init__()
        self.alpha = alpha
        self.l1_loss = nn.L1Loss()
        self.window_size = window_size
        self.sigma = sigma
        self.val_range = val_range
        self.device = device
        self.use_ms_ssim = use_ms_ssim
        self.use_edge_loss = use_edge_loss
        self.use_freq_loss = use_freq_loss
        self.edge_weight = edge_weight
        self.freq_weight = freq_weight
        
        # Register window buffer for SSIM
        self.register_buffer("window", create_window(window_size, 1, sigma, device))
        
        # Create MS-SSIM module if needed
        if use_ms_ssim:
            self.ms_ssim = MS_SSIM(window_size, sigma, val_range, device)
            
        # Edge detection kernels (Sobel)
        if use_edge_loss:
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
            self.register_buffer("sobel_x", sobel_x.repeat(1, 1, 1, 1))
            self.register_buffer("sobel_y", sobel_y.repeat(1, 1, 1, 1))
    
    def edge_loss(self, output, target):
        """Compute edge preservation loss using Sobel filters"""
        # Ensure sobel filters have the same dtype as input tensors
        sobel_x = self.sobel_x
        sobel_y = self.sobel_y
        
        if sobel_x.dtype != output.dtype or sobel_x.device != output.device:
            sobel_x = sobel_x.to(device=output.device, dtype=output.dtype)
        if sobel_y.dtype != output.dtype or sobel_y.device != output.device:
            sobel_y = sobel_y.to(device=output.device, dtype=output.dtype)
            
        # Apply Sobel filters
        output_x = F.conv2d(output, sobel_x, padding=1)
        output_y = F.conv2d(output, sobel_y, padding=1)
        target_x = F.conv2d(target, sobel_x, padding=1)
        target_y = F.conv2d(target, sobel_y, padding=1)
        
        # Compute edge magnitude
        output_mag = torch.sqrt(output_x**2 + output_y**2 + 1e-10)
        target_mag = torch.sqrt(target_x**2 + target_y**2 + 1e-10)
        
        # L1 loss on edge magnitudes
        return F.l1_loss(output_mag, target_mag)
    
    def frequency_loss(self, output, target):
        """Compute loss in frequency domain using FFT"""
        # For robustness: always use float32 for FFT operations
        # This avoids issues with half precision and non-power-of-two dimensions
        try:
            # Store original device for later
            device = output.device
            
            # Move to CPU and convert to float32 for maximum compatibility
            output_f32 = output.detach().cpu().to(torch.float32)
            target_f32 = target.detach().cpu().to(torch.float32)
            
            # Apply FFT on float32 CPU tensors
            output_fft = torch.fft.fft2(output_f32)
            target_fft = torch.fft.fft2(target_f32)
            
            # Compute magnitude spectrum
            output_mag = torch.abs(output_fft)
            target_mag = torch.abs(target_fft)
            
            # Move back to original device for loss computation
            output_mag = output_mag.to(device)
            target_mag = target_mag.to(device)
            
            # L1 loss on magnitude spectrum
            return F.l1_loss(output_mag, target_mag)
        except Exception as e:
            # Fallback in case of any FFT-related errors
            print(f"Warning: Frequency loss calculation failed: {str(e)}")
            # Return a small constant loss to avoid breaking training
            return torch.tensor(0.0, device=output.device, dtype=output.dtype)
    
    def forward(self, output, target):
        # Compute L1 loss (pixel-wise)
        l1_loss_val = self.l1_loss(output, target)
        
        # Compute structural similarity loss
        if self.use_ms_ssim:
            # Multi-scale SSIM
            ssim_val = self.ms_ssim(output, target)
        else:
            # Single-scale SSIM
            # Ensure window has the same dtype as input tensors
            window = self.window
            if window.dtype != output.dtype or window.device != output.device:
                window = window.to(device=output.device, dtype=output.dtype)
                
            ssim_val = ssim(output, target, self.window_size, self.sigma, 
                           self.val_range, self.device, window)
        
        ssim_loss = 1 - ssim_val
        
        # Base combined loss
        combined_loss = self.alpha * ssim_loss + (1 - self.alpha) * l1_loss_val
        
        # Add edge preservation loss if enabled
        if self.use_edge_loss:
            edge_loss_val = self.edge_loss(output, target)
            combined_loss += self.edge_weight * edge_loss_val
        
        # Add frequency domain loss if enabled and not using half precision
        if self.use_freq_loss and not (output.dtype == torch.float16 or output.dtype == torch.bfloat16):
            freq_loss_val = self.frequency_loss(output, target)
            combined_loss += self.freq_weight * freq_loss_val
        
        return combined_loss

class PSNR(nn.Module):
    """
    Peak Signal-to-Noise Ratio (PSNR) metric.
    Higher values indicate better image quality.
    """
    def __init__(self, max_val=1.0, epsilon=1e-10, data_range=None):
        super().__init__()
        self.max_val = max_val  # The maximum value in the expected range (1.0 for [0,1], 2.0 for [-1,1])
        self.epsilon = epsilon  # Small constant to avoid division by zero
        self.data_range = data_range  # Normalize data to [0,1] if given a range
    
    def forward(self, output, target):
        # Ensure output and target have the same shape
        if output.shape != target.shape:
            raise ValueError(f"Output shape {output.shape} doesn't match target shape {target.shape}")
        
        # For data in [-1,1] range, we need to rescale to [0,1] for proper PSNR calculation
        if self.data_range == 'normalized':
            # Convert from [-1,1] to [0,1] range
            output_rescaled = (output + 1) / 2
            target_rescaled = (target + 1) / 2
            # Use max_val as 1.0 for rescaled data
            max_val = 1.0
        else:
            # Use inputs as is
            output_rescaled = output
            target_rescaled = target
            max_val = self.max_val
            
        # Calculate MSE
        mse = F.mse_loss(output_rescaled, target_rescaled)
        
        # Add epsilon to avoid log(0) or division by zero
        psnr = 20 * torch.log10(max_val / torch.sqrt(mse + self.epsilon))
        return psnr

class ContentLoss(nn.Module):
    """
    Content loss based on feature maps from a pretrained network.
    Useful for perceptual quality assessment.
    """
    def __init__(self, feature_extractor=None):
        super().__init__()
        
        # If no feature extractor is provided, use a simple one
        if feature_extractor is None:
            # Simple feature extractor (can be replaced with VGG or other pretrained networks)
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )
            # Freeze the feature extractor
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
        else:
            self.feature_extractor = feature_extractor
    
    def forward(self, output, target):
        # Extract features
        output_features = self.feature_extractor(output)
        target_features = self.feature_extractor(target)
        
        # Calculate L2 loss between feature maps
        return F.mse_loss(output_features, target_features)

class AdaptiveLoss(nn.Module):
    """
    Adaptive loss that dynamically adjusts weights during training
    based on the relative magnitudes of different loss components.
    """
    def __init__(self, loss_modules, initial_weights=None):
        super().__init__()
        self.loss_modules = nn.ModuleList(loss_modules)
        
        # Initialize weights
        if initial_weights is None:
            initial_weights = torch.ones(len(loss_modules)) / len(loss_modules)
        
        self.register_buffer("weights", initial_weights)
        self.register_buffer("running_losses", torch.zeros(len(loss_modules)))
        self.momentum = 0.9
    
    def forward(self, output, target):
        # Compute individual losses
        losses = [module(output, target) for module in self.loss_modules]
        losses_tensor = torch.stack(losses)
        
        # Update running averages of losses
        if self.training:
            self.running_losses = self.momentum * self.running_losses + (1 - self.momentum) * losses_tensor
            
            # Normalize weights inversely proportional to running losses
            # This gives more weight to losses with smaller values
            inv_losses = 1.0 / (self.running_losses + 1e-8)
            self.weights = inv_losses / inv_losses.sum()
        
        # Compute weighted sum
        total_loss = (self.weights * losses_tensor).sum()
        return total_loss

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
        # No need to match the window to the input tensors
        # The ssim function will handle all type conversions
        return ssim(img1, img2, self.window_size, self.sigma, 
                   self.val_range, self.device, self.window)
