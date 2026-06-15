#!/usr/bin/env python3
"""
Debug script to understand why vessel cleanup is not working
"""

import sys
import json
import numpy as np
import cv2
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# Add src to path
sys.path.append('src')

def generate_vessel_mask(image_id, width, height, annotations_path):
    """Generate vessel mask from COCO annotations."""
    with open(annotations_path) as f:
        coco_data = json.load(f)
    
    # Create lookup tables
    anns_by_image = {}
    for ann in coco_data['annotations']:
        if ann['category_id'] != 26:  # Exclude stenosis
            img_id = ann['image_id'] 
            if img_id not in anns_by_image:
                anns_by_image[img_id] = []
            anns_by_image[img_id].append(ann)
    
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    
    if image_id not in anns_by_image:
        return np.array(mask)
        
    for ann in anns_by_image[image_id]:
        for segmentation in ann['segmentation']:
            if len(segmentation) >= 6:
                xy = [(segmentation[i], segmentation[i+1]) 
                      for i in range(0, len(segmentation), 2)]
                if len(xy) >= 3:
                    draw.polygon(xy, fill=255)
    return np.array(mask)

# Test parameters
image_id = 66
annotations_path = "data/arcade/syntax/val/annotations/val.json"
img_path = "data/arcade/syntax/val/images/66.png"
safety_margin = 8

print("🔬 **Debugging Vessel Cleanup Process**")

# Load original image
img = Image.open(img_path).convert('L')
img_np = np.array(img)
width, height = img.size
print(f"📏 Image size: {width}×{height}")

# Generate vessel mask
vessel_mask = generate_vessel_mask(image_id, width, height, annotations_path)
print(f"🩸 Vessel pixels: {np.sum(vessel_mask > 0)}")

# Create vessel exclusion zone
kernel_size = max(5, safety_margin + 1)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
vessel_exclusion = cv2.dilate(vessel_mask, kernel, iterations=1)
print(f"🛡️ Exclusion zone pixels: {np.sum(vessel_exclusion > 0)}")

# Test specific cell (2,3) where we saw overlap
grid_x, grid_y = 2, 3
cell_size = 64
start_x = grid_x * cell_size
start_y = grid_y * cell_size
end_x = start_x + cell_size
end_y = start_y + cell_size

print(f"\n🔍 Testing cell ({grid_x},{grid_y}): [{start_x}:{end_x}, {start_y}:{end_y}]")

# Extract cell data
img_patch = img_np[start_y:end_y, start_x:end_x]
vessel_patch = vessel_mask[start_y:end_y, start_x:end_x]
exclusion_patch = vessel_exclusion[start_y:end_y, start_x:end_x]

print(f"  📊 Patch vessel pixels: {np.sum(vessel_patch > 0)}")
print(f"  📊 Patch exclusion pixels: {np.sum(exclusion_patch > 0)}")

# Load our generated mask for this cell
try:
    mask_path = "outputs/final_clean_patches/patch_mask/66_grid_02_03.png"
    generated_mask = np.array(Image.open(mask_path))
    print(f"  📊 Generated mask pixels: {np.sum(generated_mask > 0)}")
    
    # Check overlap
    vessel_overlap = np.sum((generated_mask > 0) & (vessel_patch > 0))
    exclusion_overlap = np.sum((generated_mask > 0) & (exclusion_patch > 0))
    
    print(f"  ❗ Vessel overlap: {vessel_overlap} pixels")
    print(f"  ❗ Exclusion overlap: {exclusion_overlap} pixels")
    
    # Test manual cleanup
    print(f"\n🧹 **Testing Manual Cleanup**")
    clean_mask = generated_mask.copy()
    clean_mask[vessel_patch > 0] = 0  # Remove vessel pixels
    clean_mask[exclusion_patch > 0] = 0  # Remove exclusion pixels
    
    final_pixels = np.sum(clean_mask > 0)
    removed_pixels = np.sum(generated_mask > 0) - final_pixels
    
    print(f"  ✂️ Original pixels: {np.sum(generated_mask > 0)}")
    print(f"  ✂️ Final pixels: {final_pixels}")
    print(f"  ✂️ Removed pixels: {removed_pixels}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Top row
    axes[0,0].imshow(img_patch, cmap='gray')
    axes[0,0].set_title('Original Patch')
    axes[0,0].axis('off')
    
    axes[0,1].imshow(vessel_patch, cmap='Reds', alpha=0.8)
    axes[0,1].set_title('Vessel Mask')
    axes[0,1].axis('off')
    
    axes[0,2].imshow(exclusion_patch, cmap='Oranges', alpha=0.8)
    axes[0,2].set_title('Exclusion Zone')
    axes[0,2].axis('off')
    
    axes[0,3].imshow(generated_mask, cmap='Blues', alpha=0.8)
    axes[0,3].set_title('Generated Mask')
    axes[0,3].axis('off')
    
    # Bottom row - overlays
    axes[1,0].imshow(img_patch, cmap='gray', alpha=0.7)
    axes[1,0].imshow(vessel_patch, cmap='Reds', alpha=0.5)
    axes[1,0].imshow(generated_mask, cmap='Blues', alpha=0.5)
    axes[1,0].set_title('All Overlays')
    axes[1,0].axis('off')
    
    axes[1,1].imshow(generated_mask, cmap='Blues', alpha=0.8)
    axes[1,1].imshow(vessel_patch, cmap='Reds', alpha=0.8)
    axes[1,1].set_title('Mask + Vessels')
    axes[1,1].axis('off')
    
    axes[1,2].imshow(clean_mask, cmap='Greens', alpha=0.8)
    axes[1,2].set_title('Cleaned Mask')
    axes[1,2].axis('off')
    
    axes[1,3].imshow(img_patch, cmap='gray', alpha=0.7)
    axes[1,3].imshow(clean_mask, cmap='Greens', alpha=0.5)
    axes[1,3].set_title('Final Result')
    axes[1,3].axis('off')
    
    plt.tight_layout()
    plt.savefig('outputs/vessel_cleanup_debug.png', dpi=150)
    print("✓ Debug visualization saved: outputs/vessel_cleanup_debug.png")
    
except Exception as e:
    print(f"❌ Error loading mask: {e}")