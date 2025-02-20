import torch
import torch.nn as nn

class CNNSuperRes(nn.Module):
    def __init__(self):
        super().__init__()
        # A simple architecture with a skip connection for superresolution
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        
    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        # Add the input to the output (residual connection)
        out = out + residual
        return out
