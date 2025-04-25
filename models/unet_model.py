import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            # Set bias=False because BatchNorm has learnable bias (beta)
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

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
    """Upscaling using ConvTranspose2d then DoubleConv"""
    def __init__(self, in_ch_up, in_ch_skip, out_channels):
        """
        Args:
            in_ch_up (int): Number of channels coming from the layer below (needs upsampling).
            in_ch_skip (int): Number of channels from the skip connection.
            out_channels (int): Number of output channels for this block.
        """
        super().__init__()
        # ConvTranspose2d halves the input channels from the layer below AND doubles spatial resolution
        self.up = nn.ConvTranspose2d(in_ch_up, in_ch_up // 2, kernel_size=2, stride=2)
        # DoubleConv takes concatenated channels: skip_channels + upsampled_channels (which has in_ch_up // 2 channels)
        self.conv = DoubleConv(in_ch_skip + (in_ch_up // 2), out_channels)

    def forward(self, x1, x2):
        # x1: feature map from previous layer (e.g., bottleneck) - channels = in_ch_up
        # x2: skip connection feature map - channels = in_ch_skip
        x1 = self.up(x1) # Upsample and halve channels

        # Pad x1 to match x2 size if necessary (ConvTranspose2d might create off-by-one)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        if diffY != 0 or diffX != 0:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, # Pad left/right
                            diffY // 2, diffY - diffY // 2]) # Pad top/bottom

        # Concatenate skip connection features
        x = torch.cat([x2, x1], dim=1) # Concatenated channels: in_ch_skip + in_ch_up // 2
        return self.conv(x)

class UNetSuperRes(nn.Module):
    """
    U-Net for MRI Super-Resolution (2x spatial upscaling).
    Uses ConvTranspose2d for learned upsampling in the decoder path.
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

        # Decoder Path (4 stages) using ConvTranspose2d in Up
        # Up(in_ch_up, in_ch_skip, out_channels)
        self.up1 = Up(f*8, f*4, f*4)           # Input: f*8(1/8res), f*4(1/4res) -> Output: f*4, 1/4 res
        self.up2 = Up(f*4, f*2, f*2)           # Input: f*4(1/4res), f*2(1/2res) -> Output: f*2, 1/2 res
        self.up3 = Up(f*2, f, f)               # Input: f*2(1/2res), f(1x res)   -> Output: f,   1x res

        # Final 2x Upsampling Stage (No skip connection, just learned upsampling + refinement)
        self.final_up = nn.ConvTranspose2d(f, f // 2, kernel_size=2, stride=2) # Input: f(1x res) -> Output: f//2, 2x res
        self.final_conv = nn.Sequential(
            # Refine features after upsampling
            nn.Conv2d(f // 2, f // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(f // 2),
            nn.ReLU(inplace=True),
            # Map to output channels
            nn.Conv2d(f // 2, out_channels, kernel_size=1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                # Use Kaiming He initialization for Conv layers
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # Initialize BatchNorm weights to 1 and biases to 0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # No specific initialization needed for PixelShuffle

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

        # Final Upsampling and Output Conv
        x = self.final_up(x)    # Upsample x(1x) -> 2x res (channels: f//2)
        logits = self.final_conv(x) # Refine and map to output channels at 2x res

        # Apply clamp activation to constrain output to [0, 1]
        return torch.clamp(logits, min=0.0, max=1.0)


