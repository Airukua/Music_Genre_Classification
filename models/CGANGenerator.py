# This code inspired by the implementaion on : https://github.com/UOS-COMP6252/public/blob/main/GANs/cgan.ipynb
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes, output_dim):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 2048),
            nn.ReLU(True),
            nn.Linear(2048, output_dim),
            nn.Tanh()
        )

    def forward(self, z, labels):
        c = self.label_emb(labels)
        x = torch.cat([z, c], dim=1)
        out = self.model(x)
        return out
