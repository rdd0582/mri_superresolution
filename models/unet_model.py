import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

def icnr(w, scale=2, init_method=init.kaiming_normal_):
    """Initialise `w` (shape: out_c, in_c, k, k) with ICNR."""
    out_c, in_c, k, _ = w.shape
    # number of sub-bands
    sub_c = out_c // (scale ** 2)
    w2 = torch.zeros(sub_c, in_c, k, k)
    init_method(w2)                         # e.g. He normal
    w2 = w2.repeat_interleave(scale ** 2, dim=0)
    with torch.no_grad():
        w.copy_(w2)

class DoubleConv(nn.Module):
    """(Convolution => [GN] => LeakyReLU) * 2 with residual connection"""
    def __init__(self, in_channels, out_channels, mid_channels=None, dilation=1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        # Whether to use residual connection (only when input and output channels match)
        self.use_residual = (in_channels == out_channels)
        
        self.double_conv = nn.Sequential(
            # First conv block
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=mid_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            # Second conv block with dilation for increased receptive field
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, 
                     padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, x):
        conv_out = self.double_conv(x)
        # Add residual connection if input and output channels match
        if self.use_residual:
            return conv_out + x
        return conv_out

class Down(nn.Module):
    """Downscaling with MaxPool then DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling using bilinear interpolation + convolution to reduce checkerboard artifacts"""
    def __init__(self, in_ch_up, in_ch_skip, out_channels):
        """
        Args:
            in_ch_up (int): Number of channels coming from the layer below (needs upsampling).
            in_ch_skip (int): Number of channels from the skip connection.
            out_channels (int): Number of output channels for this block.
        """
        super().__init__()
        # Replace ConvTranspose2d with bilinear upsampling followed by 1x1 conv
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_ch_up, in_ch_up // 2, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=in_ch_up // 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        
        # DoubleConv takes concatenated channels: skip_channels + upsampled_channels
        self.conv = DoubleConv(in_ch_skip + (in_ch_up // 2), out_channels)

    def forward(self, x1, x2):
        # x1: feature map from previous layer (e.g., bottleneck) - channels = in_ch_up
        # x2: skip connection feature map - channels = in_ch_skip
        x1 = self.up(x1) # Upsample and halve channels

        # Pad x1 to match x2 size if necessary
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        if diffY != 0 or diffX != 0:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, # Pad left/right
                            diffY // 2, diffY - diffY // 2]) # Pad top/bottom

        # Concatenate skip connection features
        x = torch.cat([x2, x1], dim=1) # Concatenated channels: in_ch_skip + in_ch_up // 2
        return self.conv(x)

class PixelShuffleUp(nn.Module):
    """Upsampling module using PixelShuffle for better artifact-free upsampling"""
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        # For PixelShuffle, we need channels = out_channels * scale_factor^2
        self.conv = nn.Conv2d(in_channels, out_channels * scale_factor**2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        # Apply ICNR initialization to reduce checkerboard artifacts
        icnr(self.conv.weight, scale_factor)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class UNetSuperRes(nn.Module):
    """
    U-Net for MRI Super-Resolution (2x spatial upscaling).
    Uses dual-branch decoder with bilinear upsampling and PixelShuffle.
    Has 3 downsampling stages and 4 upsampling stages.

    Args:
        in_channels (int): Number of input image channels (e.g., 1 for grayscale).
        out_channels (int): Number of output image channels (e.g., 1 for grayscale).
        base_filters (int): Number of filters in the first convolution layer.
                            Subsequent layers scale based on this.
    """
    def __init__(self, in_channels=1, out_channels=1, base_filters=32):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_filters = base_filters
        f = base_filters

        # Encoder Path (3 stages)
        self.inc = DoubleConv(in_channels, f)     # Out: f,   1x res
        self.down1 = Down(f, f*2)                # Out: f*2, 1/2 res
        self.down2 = Down(f*2, f*4)              # Out: f*4, 1/4 res
        self.down3 = Down(f*4, f*8)              # Out: f*8, 1/8 res (bottleneck)

        # Decoder Path (4 stages) with dual-branch upsampling
        # Bilinear+Conv branch
        self.up1 = Up(f*8, f*4, f*4)           # Input: f*8(1/8res), f*4(1/4res) -> Output: f*4, 1/4 res
        self.up2 = Up(f*4, f*2, f*2)           # Input: f*4(1/4res), f*2(1/2res) -> Output: f*2, 1/2 res
        self.up3 = Up(f*2, f, f)               # Input: f*2(1/2res), f(1x res)   -> Output: f,   1x res

        # Final 2x Upsampling with dual branch
        # Branch 1: Bilinear interpolation + Conv
        self.final_up_bilinear = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(f, f // 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=f // 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        
        # Branch 2: PixelShuffle branch
        self.final_up_pixelshuffle = PixelShuffleUp(f, f // 2)
        
        # Learnable parameter for fusing bilinear and pixelshuffle outputs
        self.alpha = nn.Parameter(torch.zeros(1))
        
        # Fusion and final convolution
        self.final_conv = nn.Sequential(
            # Process combined feature maps with half the original channel count
            nn.Conv2d(f // 2, f // 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=f // 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # Map to output channels
            nn.Conv2d(f // 2, out_channels, kernel_size=1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                # Use Kaiming He initialization for Conv layers
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                # Initialize GroupNorm weights to 1 and biases to 0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)    # Features at 1x res (channels: f)
        x2 = self.down1(x1) # Features at 1/2 res (channels: f*2)
        x3 = self.down2(x2) # Features at 1/4 res (channels: f*4)
        x4 = self.down3(x3) # Bottleneck features at 1/8 res (channels: f*8)

        # Decoder
        x = self.up1(x4, x3) # Upsample x4(1/8) -> 1/4 res, concat with x3(1/4) -> Output: 1/4 res (channels: f*4)
        x = self.up2(x, x2)  # Upsample x(1/4) -> 1/2 res, concat with x2(1/2) -> Output: 1/2 res (channels: f*2)
        x = self.up3(x, x1)  # Upsample x(1/2) -> 1x res, concat with x1(1x)   -> Output: 1x res (channels: f)

        # Final Upsampling with dual branch (2x res)
        x_bilinear = self.final_up_bilinear(x)
        x_pixelshuffle = self.final_up_pixelshuffle(x)
        
        # Fuse bilinear and pixelshuffle outputs with learned weighting
        alpha_weight = torch.sigmoid(self.alpha)
        x = alpha_weight * x_bilinear + (1 - alpha_weight) * x_pixelshuffle
        
        # Final convolution followed by sigmoid to bound output in [0, 1]
        x = self.final_conv(x)
        return torch.sigmoid(x)


