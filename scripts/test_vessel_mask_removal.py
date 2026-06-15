#!/usr/bin/env python3
"""
Test script to verify vessel mask removal works correctly
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load one of our NEW generated masks and check vessel overlap
mask_path = "outputs/final_clean_patches/patch_mask/66_grid_02_03.png"
img_path = "data/arcade/syntax/val/images/66.png"

try:
    # Load mask and original image  
    mask = np.array(Image.open(mask_path))
    img = np.array(Image.open(img_path).convert('L'))
    
    print(f"Mask shape: {mask.shape}")
    print(f"Mask values: min={mask.min()}, max={mask.max()}")
    print(f"Mask pixels: {np.sum(mask > 0)}")
    
    # Extract the corresponding 64x64 patch from the original image
    # Grid (2,3) means x=2*64=128, y=3*64=192
    grid_x, grid_y = 2, 3
    start_x = grid_x * 64
    start_y = grid_y * 64
    end_x = start_x + 64
    end_y = start_y + 64
    
    img_patch = img[start_y:end_y, start_x:end_x]
    
    # Create simple visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original patch
    axes[0].imshow(img_patch, cmap='gray')
    axes[0].set_title('Original 64x64 Patch')
    axes[0].axis('off')
    
    # Mask alone
    axes[1].imshow(mask, cmap='Blues')
    axes[1].set_title('Generated Blue Mask')
    axes[1].axis('off')
    
    # Overlay to check overlap
    axes[2].imshow(img_patch, cmap='gray', alpha=0.7)
    axes[2].imshow(mask, cmap='Blues', alpha=0.5)
    axes[2].set_title('Overlay Check')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('outputs/vessel_mask_test.png', dpi=150)
    print("✓ Test visualization saved: outputs/vessel_mask_test.png")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    print("Available mask files:")
    import os
    mask_dir = "outputs/clean_grid_patches/patch_mask"
    if os.path.exists(mask_dir):
        files = sorted(os.listdir(mask_dir))[:5]
        for f in files:
            print(f"  - {f}")