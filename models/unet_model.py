import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """Basic double convolution block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downsampling block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upsampling block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle size differences
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        if diffY > 0 or diffX > 0:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """Output convolution"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return torch.sigmoid(self.conv(x))

class UNetSuperRes(nn.Module):
    """
    U-Net for MRI quality enhancement
    - Input: 320x320 1.5T MRI slice
    - Output: 320x320 enhanced MRI slice
    - Purpose: Enhance image quality and detail without changing resolution
    """
    def __init__(self, in_channels=1, out_channels=1, base_filters=32):
        super().__init__()
        
        # Initial feature extraction
        self.inc = DoubleConv(in_channels, base_filters)
        
        # Downsampling path (320 -> 160 -> 80 -> 40 -> 20)
        self.down1 = Down(base_filters, base_filters * 2)
        self.down2 = Down(base_filters * 2, base_filters * 4)
        self.down3 = Down(base_filters * 4, base_filters * 8)
        self.down4 = Down(base_filters * 8, base_filters * 16)
        
        # Upsampling path (20 -> 40 -> 80 -> 160 -> 320)
        self.up1 = Up(base_filters * 16, base_filters * 8)
        self.up2 = Up(base_filters * 8, base_filters * 4)
        self.up3 = Up(base_filters * 4, base_filters * 2)
        self.up4 = Up(base_filters * 2, base_filters)
        
        # Final output
        self.outc = OutConv(base_filters, out_channels)
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)          # 320x320
        x2 = self.down1(x1)       # 160x160
        x3 = self.down2(x2)       # 80x80
        x4 = self.down3(x3)       # 40x40
        x5 = self.down4(x4)       # 20x20
        
        # Decoder path with skip connections
        x = self.up1(x5, x4)      # 40x40
        x = self.up2(x, x3)       # 80x80
        x = self.up3(x, x2)       # 160x160
        x = self.up4(x, x1)       # 320x320
        
        return self.outc(x)
