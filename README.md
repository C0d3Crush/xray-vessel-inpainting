# CMT for Coronary X-Ray Inpainting

![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-orange.svg)
![License](https://img.shields.io/badge/license-Research-green.svg)

## TL;DR

**AI-powered vessel inpainting for X-ray angiograms using Continuously Masked Transformers.** Reconstructs vessel-free backgrounds from coronary angiography images for medical data augmentation and synthetic dataset generation.

---

## Overview

This project adapts the CMT (Continuously Masked Transformer) architecture for medical imaging, specifically targeting **vessel inpainting in coronary angiography X-rays**. Using the [ARCADE dataset](https://arcade.grand-challenge.org/), the model learns to reconstruct realistic vessel-free backgrounds, enabling:

- **Medical data augmentation** for training robust diagnostic models
- **Synthetic dataset generation** with controllable vessel patterns
- **Background reconstruction** for improved image analysis

> **Based on:** Keunsoo Ko and Chang-Su Kim, *"Continuously Masked Transformer for Image Inpainting"*, ICCV 2023

---

## Key Features

- **Medical-optimized:** Grayscale X-ray adaptation of RGB transformer architecture
- **Enhanced metrics:** PSNR, SSIM, Wasserstein Distance, RMSE tracking
- **Flexible training:** Dynamic input sizing (32-512px), multi-device support (CPU/GPU/MPS)
- **Performance optimized:** Mask caching, annotation preprocessing, smart checkpointing
- **Advanced visualization:** Adaptive vessel detection, side-by-side comparisons
- **Easy workflows:** Makefile automation for training, inference, and analysis

---

## Quick Start

### Installation

```bash
git clone https://github.com/C0d3Crush/arcade-xray-inpainting.git
cd arcade-xray-inpainting
pip install -r requirements.txt

# Verify installation
make smoke-test
```

### Basic Usage

```bash
# 1. Prepare data (optional - speeds up training)
make cache-data

# 2. Train model
make train

# 3. Generate test samples and run inference
make prepare-samples
make inference
make visualize  # Creates Input|Mask|Result comparisons
```

---

## Architecture

**Two-stage inpainting pipeline:**

```
Input X-ray → CMT Encoder → Coarse Prediction → SwinTransformer Decoder → Refined Output
     ↓              ↓                ↓                        ↓              ↓
  [256×256]    [15-layer ViT]   [Multi-scale]         [U-Net + Skip]    [256×256]
```

**Key Components:**
- **Stage 1:** ViT-based encoder with continuously masked attention (15 layers, 16 heads)
- **Stage 2:** SwinTransformer U-Net decoder with skip connections
- **Loss:** Combined L1 (masked 6× + valid 1×) + SSIM (0.5×)

**Training metrics:** PSNR, SSIM, Wasserstein Distance, RMSE evaluation

---

## Training

### Quick Training
```bash
make train  # Default: 64px, CPU, 100 epochs
```

### Custom Training
```bash
python src/train.py \
    --epochs 100 \
    --batch_size 4 \
    --input_size 256 \
    --device cuda \
    --drive_ckpt /path/to/backup  # Optional Google Drive mirroring
```

### Key Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input_size` | 64 | Image size (power of 2, min 32) |
| `--epochs` | 100 | Training epochs |
| `--batch_size` | 4 | Batch size |
| `--device` | cpu | Device (`cpu`, `cuda`, `mps`) |

---

## Inference & Visualization

### Generate Results
```bash
make prepare-samples  # Extract 5 test images from ARCADE
make inference        # Run inpainting with trained model
make visualize        # Create comparison visualizations
```

### Enhanced Visualization Features
- **Adaptive vessel detection** for optimal crop regions
- **16×16 pixel detail views** with 4× upscaling
- **Red overlay masks** for clear vessel highlighting
- **Side-by-side format:** Original | Mask | Inpainted

### Training Analysis
```bash
make plot  # Generates training curves with all metrics
```

Displays: Training Loss | Validation PSNR | Validation SSIM | Wasserstein Distance | RMSE (when available)

---

## Enhanced Metrics

**New comprehensive evaluation:**
- **PSNR** (Peak Signal-to-Noise Ratio) - Image quality
- **SSIM** (Structural Similarity Index) - Perceptual quality  
- **Wasserstein Distance** - Distribution similarity (Earth Mover's Distance)
- **RMSE** (Root Mean Square Error) - Pixel-level accuracy

All metrics logged to `checkpoints/training_log.csv` with automatic Google Drive backup.

---

## Performance Optimizations

### Data Pipeline
```bash
make cache-data  # 10× faster training via pre-computed masks
```

### Features
- **Pickle annotation caching** - Instant COCO loading
- **Smart checkpointing** - Auto-cleanup with top-K retention
- **Multi-device support** - Seamless CPU/CUDA/MPS switching
- **Memory efficient** - Dynamic batching and gradient accumulation

---

## Development

### Project Structure
```
src/              # Core training and inference code
scripts/          # Utilities (visualization, preprocessing)  
checkpoints/      # Model weights and training logs
outputs/samples/  # Test data and inference results
data/arcade/      # ARCADE dataset
Makefile          # Automated workflows
```

### Git Workflow
```bash
git pushall  # Push to GitHub + GitLab simultaneously
```

### Makefile Commands
- `make install` - Install dependencies
- `make cache-data` - Precompute masks & annotations  
- `make train` - Train model
- `make inference` - Run inference
- `make visualize` - Create comparisons
- `make plot` - Generate training plots
- `make clean` - Remove outputs

---

## Citation

```bibtex
@inproceedings{ko2023cmt,
  title={Continuously Masked Transformer for Image Inpainting},
  author={Ko, Keunsoo and Kim, Chang-Su},
  booktitle={ICCV},
  year={2023}
}
```

---

## Links

- **[ARCADE Dataset](https://arcade.grand-challenge.org/)** - Coronary angiography challenge
- **[Original CMT Paper](https://openaccess.thecvf.com/content/ICCV2023/)** - ICCV 2023
- **[GitHub](https://github.com/C0d3Crush/arcade-xray-inpainting)** - Main repository
- **[GitLab](https://gitlab.umm.uni-heidelberg.de/medphys/juergen-hesser/dzielski-fp-inpainting-for-x-rays)** - University mirror

---

