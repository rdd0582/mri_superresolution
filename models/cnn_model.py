import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    Residual block with normalization and scaling factor for stability
    """
    def __init__(self, channels, scaling_factor=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.scaling_factor = scaling_factor
        
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # Scale the residual by a small factor for training stability
        out = residual + self.scaling_factor * out
        out = self.relu(out)
        return out

class PixelShuffleUpsampler(nn.Module):
    """
    Upsampling module using pixel shuffle
    """
    def __init__(self, in_channels, scale_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * (scale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.relu(x)
        return x

class CNNSuperRes(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_features=64, num_blocks=8, scale_factor=1):
        """
        Improved CNN model for superresolution with multiple residual blocks and optional upsampling
        
        Args:
            in_channels: Number of input channels (default: 1 for grayscale)
            out_channels: Number of output channels (default: 1 for grayscale)
            num_features: Number of feature channels in hidden layers
            num_blocks: Number of residual blocks
            scale_factor: Upsampling factor (1 = no upsampling)
        """
        super().__init__()
        
        if scale_factor < 1:
            raise ValueError("Scale factor must be >= 1")
        
        self.scale_factor = scale_factor
        
        # Initial feature extraction
        self.conv_in = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(num_features)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual blocks
        res_blocks = [ResidualBlock(num_features) for _ in range(num_blocks)]
        self.res_blocks = nn.Sequential(*res_blocks)
        
        # Upsampling block (only if scale_factor > 1)
        if self.scale_factor > 1:
            self.upsampler = PixelShuffleUpsampler(num_features, scale_factor)
        else:
            self.upsampler = None
        
        # Final output layer
        self.conv_out = nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        # Store input for global residual connection
        input_img = x
        
        # Initial feature extraction
        feat = self.relu(self.bn_in(self.conv_in(x)))
        
        # Residual blocks
        res = self.res_blocks(feat)
        
        # Global residual connection
        res = res + feat
        
        # Upsampling (if applicable)
        if self.upsampler is not None:
            res = self.upsampler(res)
        
        # Final convolution
        out = self.conv_out(res)
        
        # If no upsampling, add global residual connection
        if self.scale_factor == 1:
            out = out + input_img
            
        # Constrain output to [0, 1] range
        return torch.sigmoid(out)
