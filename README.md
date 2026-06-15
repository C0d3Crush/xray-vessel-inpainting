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

## Setup

```bash
pip install -r requirements.txt
```

**Core dependencies:** PyTorch 2.2.2, timm 1.0.26, torchvision 0.17.2, scikit-image, OpenCV, einops

> **Training:** All training must be run on Google Colab (GPU required). Local machine is for development, inference, and visualization only.

---

## Quick Start

### Prepare data cache (10× faster training)
```bash
make cache-data
```

### Run inference on existing checkpoint
```bash
python scripts/patch_inference.py \
  --ckpt checkpoints/best.pth \
  --annotations data/arcade/syntax/val/annotations/val.json \
  --images data/arcade/syntax/val/images \
  --output-dir outputs/vessel_safe_patches \
  --num-images 4 --patches-per-image 3
```

### Create visual comparison
```bash
python scripts/create_training_comparison.py \
  --patch-img outputs/vessel_safe_patches/original \
  --patch-mask outputs/vessel_safe_patches/mask \
  --patch-result outputs/vessel_safe_patches/result \
  --output outputs/vessel_safe_patches/comparison.png
```

### Plot training metrics
```bash
python scripts/plot_training.py checkpoints/training_log.csv
```

---

## Training (Google Colab)

### Basic patch training
```bash
python src/train.py \
  --epochs 100 --batch_size 4 \
  --input_size 64 --patches_per_image 16 \
  --device cuda
```

### With precomputed masks (recommended)
```bash
python src/train.py \
  --train_mask data/masks_cache/train \
  --val_mask data/masks_cache/val \
  --device cuda --batch_size 16 --epochs 100
```

### Vessel-safe training (zero vessel-mask overlap)
```bash
python src/train.py --vessel_safe_training --input_size 64 --epochs 100 --device cuda
```

### Resume from checkpoint
```bash
python src/train.py --ckpt checkpoints/best.pth --epochs 150 --device cuda
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--epochs` | 100 | Training epochs |
| `--batch_size` | 4 | Batch size |
| `--input_size` | 256 | Model input size (power of 2, min 64) |
| `--lr` | 1e-4 | Learning rate |
| `--patches_per_image` | 4 | Patches extracted per image |
| `--foreground_prob` | 0.75 | Probability of vessel-biased patch sampling |
| `--vessel_safe_training` | — | Guarantee zero vessel-mask overlap |
| `--mask_weight` | 6.0 | L1 weight for masked region |
| `--ssim_weight` | 0.5 | SSIM loss weight |
| `--keep_checkpoints` | 3 | Number of top checkpoints to retain |
| `--smoke_test` | — | Quick 1-epoch test |

---

## Metrics

| Metric | Target | Description |
|---|---|---|
| PSNR | ≥ 35 dB | Peak Signal-to-Noise Ratio |
| SSIM | ≥ 0.90 | Structural Similarity Index |
| RMSE | — | Root Mean Square Error |
| Wasserstein | — | Distribution divergence (Earth Mover's Distance) |

All metrics logged to `checkpoints/training_log.csv`.

---

## Makefile Targets

```bash
make install                  # Install dependencies
make cache-data               # Precompute masks (10× speedup)
make smoke-test               # Quick pipeline test (1 epoch, CPU)
make train                    # Full training
make vessel-safe-train        # Training with vessel-safe masks
make patch-comparison         # End-to-end: prepare + inference + visualize
make inference                # Run inference on prepared samples
make plot                     # Plot training curves
make grid-workflow            # Grid mask generation + overview
make clean                    # Remove checkpoints and logs
```

---

## Scripts

| Script | Purpose |
|---|---|
| `scripts/patch_inference.py` | Extract real 64×64 patches and run inference (no resizing) |
| `scripts/generate_grid_masks.py` | Systematic 8×8 grid patches with vessel-safe masks |
| `scripts/create_grid_overview.py` | Grid visualization with coverage heatmaps |
| `scripts/cache_masks.py` | Precompute vessel masks from COCO annotations |
| `scripts/create_training_comparison.py` | Side-by-side Original / Mask / Result visualizations |
| `scripts/plot_training.py` | Training metrics plots (3 or 5 metrics, auto-detected) |
| `scripts/preprocess_coco.py` | Convert COCO JSON → pickle cache (instant loading) |
| `scripts/optimize_parameters.py` | Grid search and parameter optimization |

---

## Output Structure

```
outputs/
├── vessel_safe_patches/
│   ├── original/          # Real 64×64 patches from full images
│   ├── mask/              # Vessel masks
│   ├── result/            # Inpainted results
│   └── comparison.png     # Side-by-side visualization
├── complete_demo/
│   ├── patches/           # Grid patches
│   ├── masks/             # Grid masks
│   ├── overview/          # Grid overview visualization
│   └── comparison/        # Full comparison set
└── training_plot.png

checkpoints/
├── best.pth               # Best checkpoint (by PSNR) — not tracked in git
├── training_log.csv       # Epoch metrics
└── training_analysis.csv  # Extended analysis
```

---

## Patch-Based Evaluation

Always use real 64×64 patches — never resize full images:

```bash
# Step 1: Extract patches and run inference
python scripts/patch_inference.py \
  --ckpt checkpoints/best.pth \
  --annotations data/arcade/syntax/val/annotations/val.json \
  --images data/arcade/syntax/val/images \
  --output-dir outputs/vessel_safe_patches \
  --num-images 4 --patches-per-image 3

# Step 2: Visualize
python scripts/create_training_comparison.py \
  --patch-img outputs/vessel_safe_patches/original \
  --patch-mask outputs/vessel_safe_patches/mask \
  --patch-result outputs/vessel_safe_patches/result \
  --output outputs/vessel_safe_patches/comparison.png
```

Resizing full images to 64×64 introduces interpolation artifacts and does not reflect true model performance.

---

## Vessel-Safe Grid System

For systematic training coverage with guaranteed zero vessel overlap:

```bash
# Generate grid patches and masks
python scripts/generate_grid_masks.py \
  --annotations data/arcade/syntax/val/annotations/val.json \
  --images data/arcade/syntax/val/images \
  --output-img outputs/grid_demo/patches \
  --output-mask outputs/grid_demo/masks \
  --num-images 2 --grid-size 64

# Create overview visualization
python scripts/create_grid_overview.py \
  --annotations data/arcade/syntax/val/annotations/val.json \
  --images data/arcade/syntax/val/images \
  --output-dir outputs/grid_demo_overview \
  --num-images 1 --grid-size 64
```

The grid system uses a 6×6 inner cell layout (border patches excluded), with a 15px safety margin ensuring vessel structures never overlap with generated masks. Each patch targets 5–35% mask coverage.
