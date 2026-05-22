# CMT for Coronary X-Ray Inpainting

**AI-powered vessel inpainting for X-ray angiograms using Continuously Masked Transformers.** Reconstructs vessel-free backgrounds from coronary angiography images for medical data augmentation.

## Quick Start

```bash
git clone https://github.com/C0d3Crush/arcade-xray-inpainting.git
cd arcade-xray-inpainting
pip install -r requirements.txt

# Basic workflow
make cache-data      # Optional: 10× speed boost
make train          # Train model (64px, 100 epochs)
make prepare-samples && make inference && make visualize
```

## Key Features

- **Medical-optimized:** Grayscale X-ray adaptation of transformer architecture
- **Two-stage pipeline:** ViT encoder + SwinTransformer decoder
- **Enhanced metrics:** PSNR, SSIM, Wasserstein Distance, RMSE
- **Easy workflows:** Makefile automation for training and inference

## Training

```bash
# Quick training (CPU, 64px)
make train

# GPU training with larger input
python src/train.py --input_size 256 --batch_size 16 --device cuda --epochs 100
```

## Project Structure

```
src/              # Core training and inference code
scripts/          # Utilities (visualization, preprocessing)  
checkpoints/      # Model weights and training logs
outputs/samples/  # Test data and inference results
```

## Links

- **[ARCADE Dataset](https://arcade.grand-challenge.org/)** - Coronary angiography challenge
- **[GitHub](https://github.com/C0d3Crush/arcade-xray-inpainting)** - Main repository
- **[GitLab](https://gitlab.umm.uni-heidelberg.de/medphys/juergen-hesser/dzielski-fp-inpainting-for-x-rays)** - University mirror

> **Based on:** Keunsoo Ko and Chang-Su Kim, *"Continuously Masked Transformer for Image Inpainting"*, ICCV 2023

