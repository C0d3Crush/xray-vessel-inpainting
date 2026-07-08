# -*- coding: utf-8 -*-
import torch.nn as nn
from torch.nn.utils import spectral_norm


class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator with spectral normalization for grayscale inputs.

    Outputs a (B, 1, H/8, W/8) logit map — each logit judges realism of a
    local receptive field rather than the whole image.
    """

    def __init__(self, in_channels=1, base_channels=64):
        super().__init__()
        c = base_channels
        self.net = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, c, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(c, c * 2, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(c * 2, c * 4, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(c * 4, c * 8, 4, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(c * 8, 1, 4, stride=1, padding=1)),
        )

    def forward(self, x):
        return self.net(x)
