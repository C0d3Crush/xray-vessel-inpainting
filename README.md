# Arcade X-ray Inpainting

A CMT (Continuously Masked Transformer) model for inpainting grayscale coronary angiography X-rays from the [ARCADE dataset](https://arcade.grand-challenge.org/). Reconstructs vessel-free backgrounds by removing blood vessel structures — enabling synthetic data generation and medical image augmentation.

## Overview

**Task:** Remove vessel structures from coronary X-ray angiograms while preserving surrounding tissue.

**Applications:**
- Synthetic vessel-free background generation for data augmentation
- Downstream medical imaging model training
- Diagnostic image preprocessing

**Dataset:** ARCADE — coronary angiography images with COCO-format vessel polygon annotations.

---

## Architecture

Two-stage inpainting pipeline:

```
Grayscale X-ray [-1,1]
        ↓
  ViT Coarse Stage          (15-layer Continuously Masked Transformer)
        ↓
  SwinTransformer Refine    (U-Net decoder with skip connections)
        ↓
  Inpainted Output
```

### Stage 1 — ViT Coarse Encoder (`src/network/vit.py`)
- 15 transformer layers, 16 attention heads, hidden dim 768, MLP dim 1024
- Window-partitioned attention with continuous masking
- Patch size: `max(2, input_size // 16)`
- Input: `(B, 2, H, W)` — image + inverted mask

### Stage 2 — SwinTransformer Refine Decoder (`src/network/refine.py`)
- U-Net with SwinTransformer encoder backbone
- Input: 3 channels (2 coarse predictions + mask)
- Adaptive depth: `max(2, log2(input_size) - 4)` — scales automatically with input size
- Progressive upsampling with encoder skip connections
- Output: single-channel image with Tanh activation

### Inpainting Constraint
```python
output = (refined * mask) + img * (1 - mask)  # Preserve unmasked regions exactly
```

---

## Loss Function

```python
loss = L1(output * mask, target * mask) * 6.0       # Masked region (heavy penalty)
     + L1(output * (1-mask), target * (1-mask)) * 1.0  # Valid region (preservation)
     + (1 - SSIM(output, target)) * 0.5              # Perceptual quality
```

---

## Training

### Arguments

**Paths**

| Argument | Default | Description |
|---|---|---|
| `--train_img` | — | Training images directory |
| `--train_ann` | — | Training COCO annotation JSON |
| `--val_img` | — | Validation images directory |
| `--val_ann` | — | Validation COCO annotation JSON |
| `--train_mask` | — | Precomputed training masks directory (skips on-the-fly generation) |
| `--val_mask` | — | Precomputed validation masks directory |
| `--output_dir` | `checkpoints` | Checkpoint and log output directory |
| `--ckpt` | — | Resume training from checkpoint |

**Training**

| Argument | Default | Description |
|---|---|---|
| `--device` | `cpu` | Training device (`cpu` or `cuda`) |
| `--epochs` | 100 | Training epochs |
| `--batch_size` | 4 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--num_workers` | 0 | DataLoader worker processes |
| `--save_every` | — | Save epoch checkpoint every N epochs |
| `--keep_checkpoints` | 3 | Number of top checkpoints to retain |
| `--amp` | — | Mixed precision training (CUDA only, ~30–40% speedup) |
| `--smoke_test` | — | Quick 1-epoch pipeline verification |
| `--drive_dir` | — | Google Drive directory for checkpoint mirroring (Colab only) |

**Model & Data**

| Argument | Default | Description |
|---|---|---|
| `--input_size` | 256 | Patch size in pixels (power of 2, min 64) |
| `--patches_per_image` | 4 | Patches extracted per image per epoch |
| `--foreground_prob` | 0.75 | Probability of vessel-biased patch sampling |
| `--random_masks` | — | Use random vessel-padded masks instead of COCO annotations |
| `--mask_padding` | — | Vessel mask dilation radius (pixels) |
| `--vessel_safe_training` | — | Guarantee zero vessel-mask overlap |

**Loss**

| Argument | Default | Description |
|---|---|---|
| `--mask_weight` | 6.0 | L1 weight on masked (vessel) regions |
| `--valid_weight` | 1.0 | L1 weight on valid (background) regions |
| `--ssim_weight` | 0.5 | SSIM loss weight |

---

## Metrics

| Metric | Value |
|---|---|
| PSNR | - |
| SSIM | - |
| RMSE | - |
| Wasserstein | - |

All metrics logged to `checkpoints/training_log.csv`.

