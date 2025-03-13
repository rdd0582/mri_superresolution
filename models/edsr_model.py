import torch
import torch.nn as nn

class MeanShift(nn.Conv2d):
    """
    Mean shift layer for preprocessing and postprocessing
    """
    def __init__(self, rgb_range=1.0, rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(1, 1, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(1).view(1, 1, 1, 1) / std.view(1, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class ResidualBlock(nn.Module):
    """
    Enhanced residual block with scaling factor for stability
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
        # Scale the residual by a small factor for training stability
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
            # Use PixelShuffle for upsampling
            modules.append(nn.Conv2d(num_features, num_features * (scale ** 2), kernel_size=3, padding=1))
            modules.append(nn.PixelShuffle(scale))
            modules.append(nn.ReLU(inplace=True))
        
        super(Upsampler, self).__init__(*modules)

class EDSRSuperRes(nn.Module):
    """
    Enhanced Deep Super-Resolution Network (EDSR)
    
    Improved implementation with:
    - Proper scaling factor in residual blocks
    - Flexible input/output channels
    - Optional mean shift for preprocessing/postprocessing
    - Consistent upsampling module
    """
    def __init__(self, in_channels=1, out_channels=1, scale=1, num_res_blocks=16, 
                 num_features=64, res_scale=0.1, rgb_range=1.0, use_mean_shift=False):
        super().__init__()
        
        if scale < 1:
            raise ValueError("Scale factor must be >= 1")
        
        self.scale = scale
        self.use_mean_shift = use_mean_shift
        
        # Mean shift layers (optional)
        if use_mean_shift:
            self.sub_mean = MeanShift(rgb_range)
            self.add_mean = MeanShift(rgb_range, sign=1)
        
        # Initial feature extraction
        self.head = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)
        
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
        self.tail = nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # Store input for global residual connection if no upsampling
        input_img = x
        
        # Apply mean shift if enabled
        if self.use_mean_shift:
            x = self.sub_mean(x)
        
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
        
        # Apply mean shift if enabled
        if self.use_mean_shift:
            x = self.add_mean(x)
        
        # Add global residual connection if no upsampling
        if self.scale == 1:
            x = x + input_img
            
        return x
