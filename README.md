# CMT for Coronary Angiography X-Ray Inpainting

![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-orange.svg)
![License](https://img.shields.io/badge/license-Research-green.svg)

**Deep learning-based inpainting for coronary angiography X-rays using Continuously Masked Transformers (CMT).**

This project adapts the CMT architecture for single-channel grayscale medical imaging, specifically targeting vessel inpainting in coronary angiography images from the [ARCADE dataset](https://arcade.grand-challenge.org/). The goal is to reconstruct vessel-free backgrounds for synthetic data generation and data augmentation in medical imaging applications.

> **Based on:** Keunsoo Ko and Chang-Su Kim, *"Continuously Masked Transformer for Image Inpainting"*, ICCV 2023

---

## 🎯 Key Features

- **End-to-end CMT training** on medical imaging data (no pretraining required)
- **Grayscale adaptation** of RGB-based transformer architecture
- **Dynamic input sizing** (32px to 512px, power-of-2)
- **Multi-device support** (CPU, CUDA, Apple Silicon MPS)
- **Automated workflows** via Makefile
- **Data optimizations** (mask caching, annotation preprocessing, checkpoint rotation)
- **Comprehensive visualization** tools

---

## 📁 Project Structure

```
arcade-xray-inpainting/
├── src/                    # Core source code
│   ├── train.py           # Training script
│   ├── demo.py            # Inference script
│   ├── utils.py           # Utilities
│   └── network/           # CMT model architecture
├── scripts/                # Utility scripts
│   ├── prepare_samples.py # Sample extraction from ARCADE
│   ├── visualize_results.py # Side-by-side comparisons
│   ├── plot_training.py   # Training curves
│   └── cache_masks.py     # Mask preprocessing
├── outputs/                # Generated files
│   ├── checkpoints/       # Model weights
│   └── samples/           # Test data & results
├── data/                   # Datasets
│   └── arcade/            # ARCADE dataset
├── requirements.txt
├── Makefile               # Automated workflows
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- PyTorch 2.2+
- ARCADE dataset ([download](https://arcade.grand-challenge.org/))

### Installation

```bash
# Clone repository
git clone https://github.com/C0d3Crush/arcade-xray-inpainting.git
cd arcade-xray-inpainting

# Install dependencies
pip install -r requirements.txt

# Verify installation
make smoke-test
```

### Basic Workflow

```bash
# 1. Prepare test samples from ARCADE dataset
make prepare-samples

# 2. Run inference (requires trained model)
make inference

# 3. Create visualizations
make visualize

# 4. Plot training metrics
make plot
```

---

## 🔬 Model Architecture

The pipeline uses a two-stage architecture:

**Stage 1: Coarse Prediction**
- ViT-based encoder with continuously masked attention (15 layers, 16 heads)
- Overlapping window patch embedding
- Multi-scale coarse outputs

**Stage 2: Fine Refinement**
- SwinTransformer U-Net decoder
- Skip connections for detail preservation
- Dynamic depth scaling based on input size

**Loss Function:**
- L1 loss (masked region × 6.0)
- L1 loss (valid region × 1.0)
- SSIM loss × 0.5

**Performance:** ~37-38 dB PSNR on ARCADE validation set

---

## 💻 Training

### Quick Training

```bash
make train
```

### Custom Training

```bash
python src/train.py \
  --train_img data/arcade/syntax/train/images \
  --train_ann data/arcade/syntax/train/annotations/train.json \
  --val_img data/arcade/syntax/val/images \
  --val_ann data/arcade/syntax/val/annotations/val.json \
  --epochs 100 \
  --batch_size 4 \
  --input_size 64 \
  --device cpu \
  --output_dir outputs/checkpoints
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input_size` | 64 | Input image size (power of 2, min 32) |
| `--epochs` | 100 | Number of training epochs |
| `--batch_size` | 4 | Batch size |
| `--device` | cpu | Device (`cpu`, `cuda`, `mps`) |
| `--keep_checkpoints` | 3 | Number of periodic checkpoints to keep |
| `--ckpt` | None | Resume from checkpoint |

### Resume Training

```bash
python src/train.py --ckpt outputs/checkpoints/best.pth --epochs 150
```

---

## 🔍 Inference

### Using Makefile

```bash
make prepare-samples  # Extract test images from ARCADE
make inference        # Run inpainting
make visualize        # Create side-by-side comparisons
```

### Manual Inference

```bash
python src/demo.py \
  --ckpt outputs/checkpoints/best.pth \
  --img_path outputs/samples/test_img \
  --mask_path outputs/samples/test_mask \
  --output_path outputs/samples/results \
  --input_size 64 \
  --device cpu
```

---

## 📊 Visualization & Analysis

### Training Curves

```bash
make plot
```

Generates 3-panel plot: Training Loss | Validation PSNR | Validation SSIM

### Side-by-Side Comparisons

```bash
make visualize
```

Creates comparison images showing: **Input | Mask | Result**

### Sample Preparation

```bash
python scripts/prepare_samples.py \
  --annotations data/arcade/syntax/val/annotations/val.json \
  --images data/arcade/syntax/val/images \
  --num-samples 10 \
  --output-img outputs/samples/test_img \
  --output-mask outputs/samples/test_mask
```

---

## ⚡ Performance Optimizations

### Mask Caching

Pre-compute masks once for 10x faster data loading:

```bash
make cache-data
```

Then train with cached masks:

```bash
python src/train.py --train_mask data/masks_cache/train --val_mask data/masks_cache/val ...
```

### Annotation Preprocessing

COCO annotations are automatically cached as pickle files for instant loading (10x speedup).

### Checkpoint Rotation

Automatic cleanup of old checkpoints. Configure with `--keep_checkpoints N` (default: 3).

---

## 🔧 Development

### Git Workflow

Push to both remotes simultaneously:

```bash
git pushall  # Push to GitHub + GitLab
```

### Makefile Targets

| Command | Description |
|---------|-------------|
| `make install` | Install dependencies |
| `make smoke-test` | Quick pipeline verification |
| `make cache-data` | Precompute masks & annotations |
| `make prepare-samples` | Extract test samples from ARCADE |
| `make train` | Train model |
| `make inference` | Run inference |
| `make visualize` | Create comparisons |
| `make plot` | Plot training metrics |
| `make clean` | Remove checkpoints & logs |

---

## 🎓 Implementation Details

### Adaptations from Original CMT

- **Single-channel input:** Grayscale (1 channel) instead of RGB (3 channels)
- **Medical imaging focus:** Optimized for X-ray angiography
- **Dynamic architecture:** SwinTransformer depth auto-scales with input size
- **Multi-device support:** CPU, CUDA, and Apple Silicon (MPS)
- **Efficient data pipeline:** Mask caching and annotation preprocessing
- **Automated workflows:** Comprehensive Makefile integration

### Dataset

- **Source:** [ARCADE Challenge](https://arcade.grand-challenge.org/)
- **Format:** COCO annotations with polygon segmentations
- **Categories:** Vessel annotations (stenosis excluded)
- **Splits:** Train / Validation / Test

### Normalization

- **Images:** [-1, 1] range
- **Masks:** Binary {0, 1} (1 = vessel region to inpaint)

---

## 📝 Citation

If you use this code in your research, please cite the original CMT paper:

```bibtex
@inproceedings{ko2023cmt,
  title={Continuously Masked Transformer for Image Inpainting},
  author={Ko, Keunsoo and Kim, Chang-Su},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={},
  year={2023}
}
```

---

## 🤝 Contributing

This is a research project. For questions or collaboration opportunities, please open an issue on GitHub.

---

## 📄 License

Research use only. See repository for details.

---

## 👤 Author

**Dzielski** — Research Internship, Department of Medical Physics, Heidelberg University

---

## 🔗 Links

- [ARCADE Dataset](https://arcade.grand-challenge.org/)
- [Original CMT Paper](https://openaccess.thecvf.com/content/ICCV2023/)
- [GitHub Repository](https://github.com/C0d3Crush/arcade-xray-inpainting)
- [GitLab Repository](https://gitlab.umm.uni-heidelberg.de/medphys/juergen-hesser/dzielski-fp-inpainting-for-x-rays)

---

<p align="center">
  <i>Developed at Heidelberg University, Department of Medical Physics</i>
</p>
