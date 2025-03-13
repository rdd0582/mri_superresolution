import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ChannelAttention(nn.Module):
    """
    Channel attention module for focusing on important feature channels
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x

class SpatialAttention(nn.Module):
    """
    Spatial attention module for focusing on important spatial regions
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Generate spatial attention map
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        
        # Apply attention
        return self.sigmoid(y) * x

class DoubleConv(nn.Module):
    """
    Enhanced double convolution block with normalization options and attention
    """
    def __init__(self, in_channels, out_channels, mid_channels=None, norm_type='batch', use_attention=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        layers = []
        
        # First convolution block
        layers.append(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False))
        
        # Normalization
        if norm_type == 'batch':
            layers.append(nn.BatchNorm2d(mid_channels))
        elif norm_type == 'instance':
            layers.append(nn.InstanceNorm2d(mid_channels))
        elif norm_type == 'group':
            layers.append(nn.GroupNorm(8, mid_channels))
            
        layers.append(nn.ReLU(inplace=True))
        
        # Second convolution block
        layers.append(nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False))
        
        # Normalization
        if norm_type == 'batch':
            layers.append(nn.BatchNorm2d(out_channels))
        elif norm_type == 'instance':
            layers.append(nn.InstanceNorm2d(out_channels))
        elif norm_type == 'group':
            layers.append(nn.GroupNorm(8, out_channels))
            
        layers.append(nn.ReLU(inplace=True))
        
        self.double_conv = nn.Sequential(*layers)
        
        # Optional attention mechanism
        self.use_attention = use_attention
        if use_attention:
            self.channel_attention = ChannelAttention(out_channels)
            self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.double_conv(x)
        if self.use_attention:
            x = self.channel_attention(x)
            x = self.spatial_attention(x)
        return x

class Down(nn.Module):
    """
    Enhanced downsampling block with different downsampling options
    """
    def __init__(self, in_channels, out_channels, norm_type='batch', use_attention=False, downsample_type='maxpool'):
        super().__init__()
        
        if downsample_type == 'maxpool':
            self.down = nn.MaxPool2d(2)
        elif downsample_type == 'avgpool':
            self.down = nn.AvgPool2d(2)
        elif downsample_type == 'conv':
            self.down = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
        else:
            raise ValueError(f"Unknown downsample type: {downsample_type}")
            
        self.conv = DoubleConv(in_channels, out_channels, norm_type=norm_type, use_attention=use_attention)

    def forward(self, x):
        x = self.down(x)
        return self.conv(x)

class Up(nn.Module):
    """
    Enhanced upsampling block with different upsampling options
    """
    def __init__(self, in_channels, out_channels, norm_type='batch', use_attention=False, 
                 bilinear=True, scale_factor=2):
        super().__init__()

        # Upsampling method
        if bilinear:
            self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, 
                                  norm_type=norm_type, use_attention=use_attention)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=scale_factor, 
                                         stride=scale_factor)
            self.conv = DoubleConv(in_channels, out_channels, norm_type=norm_type, 
                                  use_attention=use_attention)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle size differences with dynamic padding
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        if diffY > 0 or diffX > 0:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
        
        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """
    Output convolution with optional activation
    """
    def __init__(self, in_channels, out_channels, activation=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = None

    def forward(self, x):
        x = self.conv(x)
        if self.activation:
            x = self.activation(x)
        return x

class PixelShuffleUpsampler(nn.Module):
    """
    Upsampling module using pixel shuffle for superresolution
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

class UNetSuperRes(nn.Module):
    """
    Enhanced U-Net architecture for MRI super-resolution with:
    - Flexible depth control
    - Attention mechanisms
    - Multiple normalization options
    - Upsampling capability for superresolution
    - Improved skip connections
    """
    def __init__(self, in_channels=1, out_channels=1, bilinear=True, base_filters=64, 
                 depth=4, norm_type='batch', use_attention=True, scale_factor=1,
                 residual_mode='add', downsample_type='maxpool'):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            bilinear: Whether to use bilinear upsampling or transposed convolutions
            base_filters: Number of base filters (doubled at each depth level)
            depth: Depth of the U-Net (number of downsampling operations)
            norm_type: Type of normalization ('batch', 'instance', 'group')
            use_attention: Whether to use attention mechanisms
            scale_factor: Upsampling factor for superresolution (1 = no upsampling)
            residual_mode: How to handle the global residual connection ('add', 'concat', 'none')
            downsample_type: Type of downsampling ('maxpool', 'avgpool', 'conv')
        """
        super().__init__()
        
        if scale_factor < 1:
            raise ValueError("Scale factor must be >= 1")
            
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        self.depth = depth
        self.scale_factor = scale_factor
        self.residual_mode = residual_mode
        
        # Initial double convolution
        self.inc = DoubleConv(in_channels, base_filters, norm_type=norm_type, use_attention=use_attention)
        
        # Dynamically create downsampling path
        self.down_path = nn.ModuleList()
        in_features = base_filters
        for i in range(depth):
            out_features = in_features * 2
            self.down_path.append(Down(in_features, out_features, norm_type=norm_type, 
                                      use_attention=use_attention, downsample_type=downsample_type))
            in_features = out_features
            
        # Dynamically create upsampling path
        self.up_path = nn.ModuleList()
        for i in range(depth):
            out_features = in_features // 2
            self.up_path.append(Up(in_features, out_features, norm_type=norm_type, 
                                  use_attention=use_attention, bilinear=bilinear))
            in_features = out_features
            
        # Final convolution
        self.outc = OutConv(base_filters, out_channels)
        
        # Optional upsampling for superresolution
        if scale_factor > 1:
            self.upsampler = PixelShuffleUpsampler(out_channels, scale_factor)
        else:
            self.upsampler = None
            
        # Additional convolution for concatenation residual mode
        if residual_mode == 'concat':
            self.concat_conv = nn.Conv2d(out_channels + in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Store input for residual connection
        input_img = x
        
        # Initial convolution
        x = self.inc(x)
        
        # Store encoder outputs for skip connections
        encoder_outputs = [x]
        
        # Encoder path
        for down in self.down_path:
            x = down(x)
            encoder_outputs.append(x)
        
        # Remove the last output (bottleneck) from skip connections
        skip_connections = encoder_outputs[:-1]
        
        # Decoder path with skip connections
        for i, up in enumerate(self.up_path):
            # Use skip connections in reverse order
            skip_idx = len(skip_connections) - i - 1
            x = up(x, skip_connections[skip_idx])
        
        # Final convolution
        x = self.outc(x)
        
        # Apply upsampling if needed
        if self.upsampler is not None:
            x = self.upsampler(x)
            # If upsampling, we don't add the residual since sizes won't match
        elif self.residual_mode == 'add':
            # Add residual connection if no upsampling
            x = x + input_img
        elif self.residual_mode == 'concat':
            # Concatenate input as an alternative to addition
            x = self.concat_conv(torch.cat([x, input_img], dim=1))
            
        return x
