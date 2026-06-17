# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CMT (Continuously Masked Transformer) adapted for inpainting grayscale X-ray angiograms from ARCADE dataset. Reconstructs vessel-free backgrounds for synthetic data generation and medical data augmentation.

**Core Task:** Train end-to-end CMT inpainting model to remove vessel structures from coronary angiography X-rays

## ⚠️ CRITICAL: Training Location
**NEVER run ANY training or smoke tests on this local machine. ALL TRAINING MUST BE DONE ON GOOGLE COLAB ONLY.**
- Local machine: Use ONLY for code development, inference with existing checkpoints, and visualization
- Google Colab: Required for ALL training operations including smoke tests (has GPU/TPU resources and proper compute environment)
- **FORBIDDEN on local machine:** `make smoke-test`, `make train`, `python src/train.py` (any training command)

## Essential Commands

### Development Workflow
```bash
pip install -r requirements.txt
make smoke-test  # Quick pipeline verification (1 epoch, CPU)

# Performance optimization (recommended before training)
make cache-data  # Precomputes masks & annotations for 10x speedup

# Standard training workflow
make train                # Train with defaults (64px, CPU, 100 epochs)
make patch-comparison     # Complete 64×64 patch workflow (prepare + inference + visualize)
make plot                 # Generate training metrics plots
```

### 64×64 Patch Comparison (Standard)
```bash
# One-command complete workflow:
make patch-comparison

# Or step by step:
make prepare-patch-samples  # Extract 64×64 patches from full-resolution images  
make inference             # Run inference on patches with best.pth
make training-comparison   # Create visualization like patch_training_comparison_epoch12
```

### Core Training Commands
```bash
# Patch training (extracts 64×64 patches from full resolution images)
python src/train.py --epochs 100 --batch_size 4 --input_size 64 --patches_per_image 16 --device cpu

# Resume from checkpoint
python src/train.py --ckpt checkpoints/best.pth --epochs 150

# GPU training with larger input
python src/train.py --input_size 256 --batch_size 16 --device cuda --epochs 100

# Training with cached masks for speed
python src/train.py --train_mask data/masks_cache/train --val_mask data/masks_cache/val
```

### Individual Script Usage
```bash
# Generate test samples
python scripts/prepare_samples.py --annotations data/arcade/syntax/val/annotations/val.json --images data/arcade/syntax/val/images --num-samples 5

# Cache masks for faster training
python scripts/cache_masks.py --annotations data/arcade/syntax/train/annotations/train.json --images data/arcade/syntax/train/images --output data/masks_cache/train

# Inference
python src/demo.py --ckpt checkpoints/best.pth --img_path outputs/samples/test_img --mask_path outputs/samples/test_mask --output_path outputs/samples/results

# Visualization  
python scripts/create_training_comparison.py --patch-img outputs/samples/patch_img --patch-mask outputs/samples/patch_mask --patch-result outputs/samples/patch_results --output outputs/samples/patch_training_comparison.png

# Plot training curves (supports 3 or 5 metrics)
python scripts/plot_training.py checkpoints/training_log.csv
```

## Architecture Overview

**Two-Stage Inpainting Pipeline:**
```
Input X-ray → ArcadeDataset → Inpaint Model → InpaintingLoss
                    ↓              ↓               ↓
              [Mask Generation] [Coarse + Fine] [L1 + SSIM]
```

### Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| **ArcadeDataset** | `src/dataset.py` | COCO annotation parsing, mask generation, grayscale normalization [-1,1] |
| **Inpaint** | `src/network/network_pro.py:6-23` | Main model: ViT coarse stage + SwinTransformer refinement |
| **ViT (Coarse)** | `src/network/vit.py` | 15-layer continuously masked transformer encoder with window partitioning |
| **Refine (Fine)** | `src/network/refine.py:16-85` | SwinTransformer U-Net decoder with adaptive depth scaling |
| **Loss** | `src/losses.py:22` | Combined L1 (masked×6.0 + valid×1.0) + SSIM×0.5 |
| **Trainer** | `src/trainer.py` | Training loop, validation, checkpoint management |

### Data Flow Architecture
1. **ArcadeDataset** loads grayscale X-rays + generates masks from COCO vessel polygons (excluding stenosis category 26)
2. **ViT coarse stage** processes `img * (1 - mask)` through continuously masked attention with window partitioning
3. **SwinTransformer refine stage** takes concatenated coarse outputs + mask, upsamples with skip connections
4. **Inpainting constraint:** `gen = (gen * mask) + img * (1 - mask)` preserves unmasked regions
5. **InpaintingLoss** combines reconstruction quality (L1) + perceptual similarity (SSIM)

## File Structure

```
src/                     # Core implementation
├── train.py            # Entry point: argument parsing, orchestrates training
├── dataset.py          # ArcadeDataset: COCO parsing, mask generation, patch extraction
├── losses.py           # InpaintingLoss: L1 (masked×6.0 + valid×1.0) + SSIM×0.5
├── trainer.py          # Training loop, validation, checkpoint management
├── demo.py             # Inference script
├── utils.py            # Checkpoint loading, PSNR/SSIM/Wasserstein/RMSE metrics
└── network/
    ├── network_pro.py  # Main Inpaint model combining ViT + SwinTransformer
    ├── vit.py         # Continuously Masked Transformer encoder
    ├── refine.py      # SwinTransformer U-Net decoder
    └── swin.py        # SwinTransformer utilities

scripts/                # Preprocessing and analysis utilities
├── cache_masks.py     # Precompute vessel masks from COCO annotations
├── prepare_samples.py # Extract test images from ARCADE dataset
├── visualize_results.py # Create side-by-side comparison visualizations
├── plot_training.py   # Plot training metrics (adaptive 3 or 5 metrics)
├── preprocess_coco.py # Convert COCO JSON to pickle for faster loading

checkpoints/           # Model weights and logs (NOT outputs/checkpoints/)
├── best.pth          # Best validation PSNR checkpoint
├── epoch_*.pth       # Periodic checkpoints
└── training_log.csv  # Training metrics: epoch,train_loss,val_psnr,val_ssim,val_wasserstein,val_rmse

outputs/samples/       # Inference results and test data
├── test_img/         # Test input images
├── test_mask/        # Generated vessel masks
├── results/          # Inpainting results
└── comparisons/      # Side-by-side visualizations
```

## Critical Implementation Details

### Training Mode
**Patch Training:** Extract random 64×64 patches from full-resolution images
- Preserves original image detail and vessel boundaries
- Multiplies dataset size (e.g., 1000 images → 16,000 patches with `--patches_per_image 16`)
- Better learning from high-resolution data
- Default `foreground_prob=0.75` biases patch selection toward vessel-rich regions

### Mask Generation Modes
1. **COCO Annotations** (default): Vessel polygons rasterized to binary masks
2. **Precomputed Masks** (`--train_mask path`): Load cached masks for speed
3. **Random Masks** (`--random_masks`): Random vessel-padded masks for training diversity

### Input Requirements
- **Image format:** Grayscale PNG, normalized to [-1, 1] range via `(img / 255.0) * 2.0 - 1.0`
- **Mask format:** Binary {0, 1}, where 1 = vessel region to inpaint (255 → 1.0 in preprocessing)
- **Input size:** Power of 2, minimum 32px for Inpaint model, minimum 64px for Refine stage
- **Patch size:** Auto-calculated as `max(2, input_size // 16)` for ViT encoder
- **SwinTransformer depth:** Auto-scales as `max(2, int(log2(input_size)) - 4)`
- **Dataset:** ARCADE COCO format with vessel polygon annotations

### Device Support
- **CPU:** Full support, default for development (CUDA disabled in demo.py)
- **CUDA:** GPU acceleration for training
- **MPS:** Apple Silicon support

### Performance Optimizations
- **Mask caching:** `make cache-data` precomputes masks → 10x faster training
- **COCO preprocessing:** Auto-detects `.pkl` cache alongside `.json` → instant loading
- **Checkpoint rotation:** Auto-cleanup, keeps top-K checkpoints (`--keep_checkpoints`)
- **Adaptive plotting:** Visualizes 3 legacy metrics or 5 enhanced metrics automatically

## Enhanced Metrics System
Training now tracks comprehensive evaluation:
- **PSNR** - Peak Signal-to-Noise Ratio (image quality)
- **SSIM** - Structural Similarity Index (perceptual quality)  
- **Wasserstein Distance** - Distribution similarity (Earth Mover's Distance)
- **RMSE** - Root Mean Square Error (pixel-level accuracy)

All logged to `checkpoints/training_log.csv` with optional Google Drive mirroring.

## Development Workflow

### Git & Versioning
**Dual-remote setup:**
- `origin` → GitHub (main branch)  
- `gitlab` → Uni Heidelberg GitLab (lukas/main, protected)

**Conventional Commit Types:**
- `feat:` → minor version bump (1.0.0 → 1.1.0)
- `fix:` → patch version bump (1.0.0 → 1.0.1)  
- `docs:`, `style:`, `refactor:`, `test:`, `chore:` → patch version bump

**Commit Guidelines:**
- **NO Co-Authored-By:** Never include "Co-Authored-By: Claude <noreply@anthropic.com>" in commit messages
- **NO Generated with Claude Code:** Never include "🤖 Generated with [Claude Code](https://claude.ai/code)" in commit messages
- **Clean commit messages:** Use only conventional commit format without Claude attribution

## Important Notes

- **Patch training recommended:** Use `--patches_per_image 16` for best results
- **Checkpoint paths:** Use `checkpoints/` not `outputs/checkpoints/` (corrected from legacy docs)
- **Stenosis exclusion:** COCO category ID 26 (stenosis) automatically filtered from vessel masks in `ArcadeDataset.__init__`
- **Input size scaling:** SwinTransformer depth auto-adjusts: `max(2, int(log2(input_size)) - 4)`
- **Memory management:** Larger input sizes require proportionally larger batch sizes for GPU memory
- **Reproducibility:** Set seeds for deterministic training and mask generation

## Model Architecture Details

### ViT Coarse Stage (`src/network/vit.py`)
- **Transformer layers:** 15 layers with 16 attention heads
- **Hidden dimension:** 768
- **MLP dimension:** 1024  
- **Window partitioning:** Overlapping windows with continuous masking
- **Output:** Multi-scale coarse predictions for refinement stage

### SwinTransformer Refine Stage (`src/network/refine.py`)
- **Input channels:** Coarse predictions (2) + mask (1) = 3 channels
- **Architecture:** U-Net with SwinTransformer encoder and conv decoder
- **Adaptive depth:** Automatically scales based on input size to maintain minimum 4×4 spatial resolution
- **Skip connections:** Between encoder stages and decoder upsampling blocks

### Loss Function (`src/losses.py:22`)
```python
loss_mask  = l1(output * mask,       target * mask)       * mask_weight   # default 6.0
loss_valid = l1(output * (1 - mask), target * (1 - mask)) * valid_weight  # default 1.0
loss_ssim  = 1 - ssim(output, target)                     * ssim_weight   # default 0.5
total_loss = loss_mask + loss_valid + loss_ssim
```
- **Weighted L1:** 6× penalty on masked regions, 1× on valid regions
- **SSIM penalty:** Structural similarity with 0.5× weight
- **Data range:** [-1, 1] normalized inputs

## 🎯 CRITICAL: How to Create Proper Patch Comparisons

### The ONLY Correct Approach for Real Patch Evaluation

**Step 1: Extract Real 64×64 Patches + Run Inference**
```bash
python scripts/patch_inference.py \
  --ckpt checkpoints/best.pth \
  --annotations data/arcade/syntax/val/annotations/val.json \
  --images data/arcade/syntax/val/images \
  --output-dir outputs/vessel_safe_patches \
  --num-images 4 \
  --patches-per-image 3
```

**Step 2: Create Visual Comparison**
```bash
python scripts/create_training_comparison.py \
  --patch-img outputs/vessel_safe_patches/original \
  --patch-mask outputs/vessel_safe_patches/mask \
  --patch-result outputs/vessel_safe_patches/result \
  --output outputs/vessel_safe_patches/comparison.png \
  --title "Real 64x64 Patch Results"
```

### ⚠️ Why This Approach is Critical:
- **Real Patches**: Extracts actual 64×64 regions from full images (no resizing)
- **Native Resolution**: Model processes patches at training resolution  
- **No Artifacts**: Avoids quality loss from interpolation operations
- **True Performance**: Shows actual model capability on patch-sized data
- **Grid-Compatible**: Works with vessel-safe grid training methodology

### 🚫 NEVER Use These (Wrong Approaches):
- `scripts/prepare_samples.py` (resizes full images to 64×64)
- `src/demo.py` on full images (resize → inference → resize back)
- Any workflow involving image resizing operations

### Output Structure:
```
outputs/vessel_safe_patches/
├── original/     # Real 64×64 patches from full images
├── mask/         # Vessel masks for each patch
├── result/       # Inpainted results at native resolution
└── comparison.png # Side-by-side: Original | Mask | Result
```

This is the definitive method for evaluating patch-based inpainting models and must be used for all quality assessments.

## 🎯 CRITICAL: How to Generate Vessel-Safe Grid Overview

### The ONLY Correct Approach for Grid Mask Generation Visualization

**Step 1: Generate Grid Patches and Masks**
```bash
python scripts/generate_grid_masks.py \
  --annotations data/arcade/syntax/val/annotations/val.json \
  --images data/arcade/syntax/val/images \
  --output-img outputs/grid_demo/patches \
  --output-mask outputs/grid_demo/masks \
  --num-images 2 \
  --grid-size 64
```

**Step 2: Create Grid Overview Visualization**
```bash
python scripts/create_grid_overview.py \
  --annotations data/arcade/syntax/val/annotations/val.json \
  --images data/arcade/syntax/val/images \
  --output-dir outputs/grid_demo_overview \
  --num-images 1 \
  --grid-size 64
```

### ⚠️ Why This Approach is Critical:
- **6×6 Inner Grid**: Uses systematic 6×6 inner cells (excludes border patches)
- **Vessel-Safe Generation**: Shows zero vessel-mask overlap (red vessels, blue masks never touch)
- **Guaranteed Masks**: Every inner patch has training signal (90%+ success rate)
- **Systematic Coverage**: Complete spatial coverage, not random sampling
- **Quality Control**: 5-35% coverage per patch with diverse shapes
- **Perfect Separation**: 15px safety margin ensures vessel exclusion

### Output Visualization Shows:
```
Grid Overview Contains:
├── Complete 8×8 Grid        # Shows full image grid layout
├── Vessel Structures (Red)  # COCO vessel annotations  
├── Generated Masks (Blue)   # Vessel-safe background masks
├── Combined View            # Perfect separation demonstration
├── Coverage Heatmap         # Mask distribution analysis
└── Technical Stats          # Success rates and parameters
```

### 🚫 NEVER Use Alternative Methods:
- Random patch sampling without grid structure
- Mask generation that overlaps with vessel regions
- Border patches (quality issues)
- Manual mask creation approaches

This is the definitive vessel-safe grid methodology that ensures systematic coverage, zero vessel overlap, and guaranteed training signal for every patch. Must be used for all mask generation demonstrations and quality verification.