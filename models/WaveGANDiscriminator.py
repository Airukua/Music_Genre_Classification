#This block of code is orginally from WaveGAN paper that publish their work in https://github.com/chrisdonahue/wavegan/blob/master/wavegan.py
import torch.nn as nn
import torch 
import torch.nn.functional as F

class PhaseShuffle(nn.Module):
    def __init__(self, rad, pad_type='reflect'):
        super(PhaseShuffle, self).__init__()
        self.rad = rad
        self.pad_type = pad_type

    def forward(self, x):
        if self.rad == 0:
            return x
        b, c, t = x.size()
        phase = torch.randint(-self.rad, self.rad + 1, (1,)).item()
        pad_l = max(phase, 0)
        pad_r = max(-phase, 0)
        x = F.pad(x, (pad_l, pad_r), mode=self.pad_type)
        phase_start = pad_r
        x = x[:, :, phase_start:phase_start + t]
        return x

class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, use_phaseshuffle=False):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=4, padding=kernel_size // 2
        )
        self.lrelu = nn.LeakyReLU(0.2)
        self.phaseshuffle = PhaseShuffle(rad=2) if use_phaseshuffle else None

    def forward(self, x):
        x = self.conv(x)
        x = self.lrelu(x)
        if self.phaseshuffle is not None:
            x = self.phaseshuffle(x)
        return x

#This block of code is orginally from WaveGAN paper that publish their work in https://github.com/chrisdonahue/wavegan/blob/master/wavegan.py
class Discriminator(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.net = nn.Sequential(
            DBlock(1, dim, 25, use_phaseshuffle=True), #0 upsample
            DBlock(dim, dim*2, 25, use_phaseshuffle=True), #1 upsample
            DBlock(dim*2, dim*4, 25, use_phaseshuffle=True), #2 upsample
            DBlock(dim*4, dim*8, 25, use_phaseshuffle=False), #3 upsample
            DBlock(dim*8, dim*16, 25, use_phaseshuffle=False), #4 upsample
            DBlock(dim*16, dim*32, 25, use_phaseshuffle=False), #5 upsample
        )
        self.final = nn.Linear(dim * 32 * 16, 1)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        x = self.final(x)
        return x