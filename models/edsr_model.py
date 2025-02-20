import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return x + out

class EDSRSuperRes(nn.Module):
    def __init__(self, scale=1, num_res_blocks=8, num_features=64):
        """
        EDSR architecture for single-channel (grayscale) images.
        If scale > 1, the network upsamples the input by the given scale factor.
        If scale == 1, the upsampling block is skipped.
        """
        super().__init__()
        
        if scale < 1:
            raise ValueError("Scale factor must be >= 1")
        self.scale = scale

        # Initial feature extraction
        self.conv1 = nn.Conv2d(1, num_features, kernel_size=3, padding=1)

        # Series of residual blocks
        res_blocks = [ResidualBlock(num_features) for _ in range(num_res_blocks)]
        self.res_blocks = nn.Sequential(*res_blocks)

        # Convolution after residual blocks
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

        # Upsampling layers (only used if scale > 1)
        if self.scale > 1:
            self.upconv = nn.Conv2d(num_features, num_features * (scale ** 2), kernel_size=3, padding=1)
            self.pixel_shuffle = nn.PixelShuffle(scale)
        else:
            self.upconv = None
            self.pixel_shuffle = None

        # Final output convolution
        self.conv3 = nn.Conv2d(num_features, 1, kernel_size=3, padding=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out1 = self.relu(self.conv1(x))
        out = self.res_blocks(out1)
        out = self.conv2(out) + out1
        if self.scale > 1:
            out = self.relu(self.upconv(out))
            out = self.pixel_shuffle(out)
        out = self.conv3(out)
        return out
