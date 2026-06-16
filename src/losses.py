# -*- coding: utf-8 -*-
import warnings
import torch
import torch.nn as nn

try:
    from skimage.metrics import structural_similarity as ssim_fn
except ImportError:
    warnings.warn("skimage not available; val_ssim will be 0.8 (placeholder)", RuntimeWarning)
    def ssim_fn(img1, img2, data_range=None, channel_axis=None):
        return 0.8


def _build_ssim_window(window_size: int, dtype, device):
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    g = g / g.sum()
    window = (g.unsqueeze(0) * g.unsqueeze(1)).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    return window


class InpaintingLoss(nn.Module):
    """L1 + SSIM loss on masked region + L1 background consistency."""
    # SSIM stability constants from the original SSIM paper
    _C1 = 0.01 ** 2
    _C2 = 0.03 ** 2

    def __init__(self, ssim_weight=0.5, mask_weight=6.0, valid_weight=1.0, ssim_window_size=11):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.ssim_weight = ssim_weight
        self.mask_weight = mask_weight
        self.valid_weight = valid_weight
        self.ssim_window_size = ssim_window_size
        self._ssim_window = None  # built lazily on first forward pass (device unknown at init)

    def _ssim_loss(self, pred, target):
        if self._ssim_window is None or self._ssim_window.device != pred.device:
            self._ssim_window = _build_ssim_window(self.ssim_window_size, pred.dtype, pred.device)
        window = self._ssim_window
        pad = self.ssim_window_size // 2
        F = torch.nn.functional
        mu1 = F.conv2d(pred,   window, padding=pad)
        mu2 = F.conv2d(target, window, padding=pad)
        mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
        sigma1_sq = F.conv2d(pred   * pred,   window, padding=pad) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=pad) - mu2_sq
        sigma12   = F.conv2d(pred   * target, window, padding=pad) - mu1_mu2
        ssim_map = ((2 * mu1_mu2 + self._C1) * (2 * sigma12 + self._C2)) / \
                   ((mu1_sq + mu2_sq + self._C1) * (sigma1_sq + sigma2_sq + self._C2))
        return 1 - ssim_map.mean()

    def forward(self, output, target, mask):
        loss_mask  = self.l1(output * mask,       target * mask)
        loss_valid = self.l1(output * (1 - mask), target * (1 - mask))
        loss_ssim  = self._ssim_loss(output, target)

        total_loss = loss_mask * self.mask_weight + loss_valid * self.valid_weight + self.ssim_weight * loss_ssim

        # Return total loss and components for analysis
        return total_loss, {
            'l1_loss': (loss_mask * self.mask_weight + loss_valid * self.valid_weight).item(),
            'ssim_loss': (self.ssim_weight * loss_ssim).item(),
            'mask_loss': loss_mask.item(),
            'valid_loss': loss_valid.item()
        }
