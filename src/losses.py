# -*- coding: utf-8 -*-
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

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


class PerceptualLoss(nn.Module):
    """VGG16 feature matching loss for texture supervision (grayscale → RGB adapter included)."""

    _VGG_LAYERS = {'relu1_2': 4, 'relu2_2': 9, 'relu3_3': 16}
    _MEAN = [0.485, 0.456, 0.406]
    _STD  = [0.229, 0.224, 0.225]

    def __init__(self, layer_weights=None):
        super().__init__()
        self.layer_weights = layer_weights or {'relu1_2': 1.0, 'relu2_2': 1.0, 'relu3_3': 1.0}

        vgg_features = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT).features
        self.slices = nn.ModuleDict()
        prev = 0
        for name, idx in sorted(self._VGG_LAYERS.items(), key=lambda x: x[1]):
            if name in self.layer_weights:
                self.slices[name] = nn.Sequential(*list(vgg_features.children())[prev:idx + 1])
                prev = idx + 1

        for param in self.parameters():
            param.requires_grad = False

        mean = torch.tensor(self._MEAN).view(1, 3, 1, 1)
        std  = torch.tensor(self._STD).view(1, 3, 1, 1)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def _preprocess(self, x):
        """(B,1,H,W) in [-1,1] → (B,3,H,W) ImageNet-normalised."""
        x = (x + 1.0) / 2.0
        x = x.repeat(1, 3, 1, 1)
        return (x - self.mean) / self.std

    def forward(self, output, target):
        out_f = self._preprocess(output)
        tgt_f = self._preprocess(target)
        loss = torch.tensor(0.0, device=output.device)
        for name, slice_net in self.slices.items():
            out_f = slice_net(out_f)
            tgt_f = slice_net(tgt_f)
            loss = loss + self.layer_weights[name] * F.l1_loss(out_f, tgt_f.detach())
        return loss


def discriminator_hinge_loss(real_logits, fake_logits):
    """Hinge loss for the discriminator: push real logits above +1, fake below -1."""
    return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()


def generator_hinge_loss(fake_logits):
    """Hinge loss for the generator: maximise discriminator output on fakes."""
    return -fake_logits.mean()


class InpaintingLoss(nn.Module):
    """L1 + SSIM loss on masked region.

    No valid-region term: the network composites gen = gen*mask + img*(1-mask),
    so unmasked pixels always equal the target and such a loss would be zero.
    """
    # SSIM stability constants from the original SSIM paper
    _C1 = 0.01 ** 2
    _C2 = 0.03 ** 2

    def __init__(self, ssim_weight=0.5, mask_weight=6.0,
                 perceptual_weight=0.0, ssim_window_size=11):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.ssim_weight = ssim_weight
        self.mask_weight = mask_weight
        self.perceptual_weight = perceptual_weight
        self.ssim_window_size = ssim_window_size
        self._ssim_window = None  # built lazily on first forward pass (device unknown at init)
        if perceptual_weight > 0:
            self.perceptual = PerceptualLoss()

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
        loss_mask  = self.l1(output * mask, target * mask)
        loss_ssim  = self._ssim_loss(output, target)

        total_loss = loss_mask * self.mask_weight + self.ssim_weight * loss_ssim

        components = {
            'l1_loss': (loss_mask * self.mask_weight).item(),
            'ssim_loss': (self.ssim_weight * loss_ssim).item(),
            'mask_loss': loss_mask.item(),
        }

        if self.perceptual_weight > 0:
            loss_perceptual = self.perceptual(output, target)
            total_loss = total_loss + self.perceptual_weight * loss_perceptual
            components['perceptual_loss'] = loss_perceptual.item()

        return total_loss, components
