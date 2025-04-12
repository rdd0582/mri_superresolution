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
    """Upscaling using Bilinear Interpolation then DoubleConv (No Attention)"""
    def __init__(self, in_ch_up, in_ch_skip, out_channels):
        """
        Args:
            in_ch_up (int): Number of channels coming from the layer below (needs upsampling).
            in_ch_skip (int): Number of channels from the skip connection.
            out_channels (int): Number of output channels for this block.
        """
        super().__init__()

        # Upsampling layer using Bilinear interpolation
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # Convolution after upsampling to potentially refine features
        # Input: in_ch_up, Output: in_ch_up
        self.conv_after_up = nn.Conv2d(in_ch_up, in_ch_up, kernel_size=1)

        # After concatenation, channels will be in_ch_skip + in_ch_up
        self.conv = DoubleConv(in_ch_skip + in_ch_up, out_channels)

    def forward(self, x1, x2):
        # x1: feature map from previous layer (e.g., bottleneck) - channels = in_ch_up
        # x2: skip connection feature map - channels = in_ch_skip

        # Apply upsampling and convolution
        x1 = self.up(x1)
        x1 = self.conv_after_up(x1)

        # Pad x1 to match x2 size if necessary
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        if diffY > 0 or diffX > 0:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
        elif diffY < 0 or diffX < 0:
             x2 = x2[:, :, abs(diffY) // 2 : x2.size()[2] - (abs(diffY) - abs(diffY) // 2),
                       abs(diffX) // 2 : x2.size()[3] - (abs(diffX) - abs(diffX) // 2)]

        # Directly concatenate the original skip connection features (x2)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """Final 1x1 convolution to map features to output channels"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Return logits; apply activation in the main model's forward pass
        return self.conv(x)

class UNetSuperRes(nn.Module):
    """
    U-Net for MRI quality enhancement (preserves resolution).
    Uses Bilinear Upsampling + Conv in the decoder path (No Attention).

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
        f = base_filters # Shortcut for filter count

        # Encoder Path
        self.inc = DoubleConv(in_channels, f)            # Output channels: f
        self.down1 = Down(f, f*2)                       # Output channels: f*2
        self.down2 = Down(f*2, f*4)                     # Output channels: f*4
        self.down3 = Down(f*4, f*8)                     # Output channels: f*8
        self.down4 = Down(f*8, f*16)                    # Bottleneck channels: f*16

        # Decoder Path
        # Arguments for Up: (in_ch_up, in_ch_skip, out_channels)
        self.up1 = Up(f*16, f*8, f*8)         # Input: f*16 (up), f*8 (skip) -> Output: f*8
        self.up2 = Up(f*8, f*4, f*4)          # Input: f*8 (up), f*4 (skip) -> Output: f*4
        self.up3 = Up(f*4, f*2, f*2)          # Input: f*4 (up), f*2 (skip) -> Output: f*2
        self.up4 = Up(f*2, f, f)              # Input: f*2 (up), f (skip) -> Output: f

        # Final Output Layer
        self.outc = OutConv(f, out_channels)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # Bias is already handled (set to False or initialized to 0 if present)
                if m.bias is not None and m.bias.requires_grad: # Check if bias exists and requires grad
                     nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # Initialize PixelShuffle's preceding Conv2d layers with care
            elif isinstance(m, nn.PixelShuffle):
                # PixelShuffle itself has no weights, but we ensure preceding Conv2d is properly initialized
                pass

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4) # Bottleneck features

        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Final convolution and activation
        logits = self.outc(x)
        # Apply sigmoid activation to constrain output to [0, 1]
        return torch.sigmoid(logits)


