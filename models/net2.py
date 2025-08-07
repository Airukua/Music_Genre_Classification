#This code is implemented from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
#It also inspired by https://github.com/UOS-COMP6252/public/blob/main/lecture5/conv.ipynb
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)

class PoolBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x):
        return self.block(x)

class Net2(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(3, 32),     # conv1
            PoolBlock(32, 32),    # conv2 + MaxPool1
            ConvBlock(32, 64),    # conv3
            PoolBlock(64, 128),   # conv4 + MaxPool2
            nn.Flatten(),
            nn.Linear(128 * 45 * 45, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.net(x)
