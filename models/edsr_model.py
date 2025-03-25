import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    Simplified residual block with scaling factor for stability
    """
    def __init__(self, num_features, scaling_factor=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return residual + self.scaling_factor * out

class Upsampler(nn.Sequential):
    """
    Upsampling module using pixel shuffle
    """
    def __init__(self, scale, num_features):
        modules = []
        if scale == 1:
            pass
        else:
            modules.append(nn.Conv2d(num_features, num_features * (scale ** 2), kernel_size=3, padding=1))
            modules.append(nn.PixelShuffle(scale))
            modules.append(nn.ReLU(inplace=True))
        
        super(Upsampler, self).__init__(*modules)

class EDSRSuperRes(nn.Module):
    """
    Simplified EDSR model for grayscale MRI images
    
    Optimized for grayscale MRI processing with:
    - Fixed single channel input/output
    - Reduced default number of residual blocks for efficiency
    - Removed mean shift operations (unnecessary for MRI data)
    - Retained upsampling capabilities for super-resolution
    """
    def __init__(self, scale=1, num_res_blocks=8, num_features=64, res_scale=0.1):
        super().__init__()
        
        if scale < 1:
            raise ValueError("Scale factor must be >= 1")
        
        self.scale = scale
        
        # Initial feature extraction
        self.head = nn.Conv2d(1, num_features, kernel_size=3, padding=1)
        
        # Residual blocks
        modules_body = [
            ResidualBlock(num_features, scaling_factor=res_scale) 
            for _ in range(num_res_blocks)
        ]
        modules_body.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1))
        self.body = nn.Sequential(*modules_body)
        
        # Upsampling module
        self.upsampler = Upsampler(scale, num_features)
        
        # Final output convolution
        self.tail = nn.Conv2d(num_features, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # Store input for global residual connection if no upsampling
        input_img = x
        
        # Head
        x = self.head(x)
        head_features = x
        
        # Body (residual blocks)
        x = self.body(x)
        
        # Global residual connection in feature space
        x = x + head_features
        
        # Upsampling
        x = self.upsampler(x)
        
        # Final convolution
        x = self.tail(x)
        
        # Add global residual connection if no upsampling
        if self.scale == 1:
            x = x + input_img
            
        # Constrain output to [0, 1] range
        return torch.sigmoid(x)
