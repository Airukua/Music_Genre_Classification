#This block of code is orginally from WaveGAN paper that publish their work in https://github.com/chrisdonahue/wavegan/blob/master/wavegan.py
#This block of code utilize their default parameter and converted tensorflow to pytorch only difference is I use slice 65536
import torch.nn as nn

class Conv2DTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(Conv2DTranspose, self).__init__()
        self.conv = nn.ConvTranspose1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=11,
            output_padding=1
        )

    def forward(self,x):
        return self.conv(x)

class z_project(nn.Module):
    def __init__(self, z=100, dim_mul=32, dim=64, batch_size=32):
        super(z_project, self).__init__()
        self.batch_size = batch_size
        self.dim_mul = dim_mul
        self.dim = dim

        self.z_project = nn.Sequential(
            nn.Linear(z, 4 * 4 * dim * dim_mul),
            nn.Unflatten(1, (dim * dim_mul, 16)),
            nn.ReLU()
        )

    def forward(self, x):
        return self.z_project(x)

#This block of code is orginally from WaveGAN paper that publish their work in https://github.com/chrisdonahue/wavegan/blob/master/wavegan.py
class Generator(nn.Module):
    def __init__(self, dim_mul=32, dim=64):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            z_project(),
            Conv2DTranspose(dim_mul*dim, 16*dim, 25, 4), #0 upsample
            nn.ReLU(),
            Conv2DTranspose(16*dim, 8*dim, 25, 4), #1 upsample
            nn.ReLU(),
            Conv2DTranspose(8*dim, 4*dim, 25, 4), #2 upsample
            nn.ReLU(),
            Conv2DTranspose(4*dim,2*dim, 25, 4),#3 upsample
            nn.ReLU(),
            Conv2DTranspose(2*dim,dim, 25, 4),#3 upsample
            nn.ReLU(),
            Conv2DTranspose(dim,1, 25, 4),
            nn.Tanh()
        )
    def forward(self,x):
        return self.net(x)