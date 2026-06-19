#!/usr/bin/env python3
"""
Extract 64x64 patches for proper visualization of patch training results.
This shows what the model actually sees during training and inference.
"""

import argparse, logging, os, cv2, glob
import numpy as np
from scipy.ndimage import binary_dilation

logger = logging.getLogger(__name__)

def add_vessel_padding(mask, padding_radius=3):
    """Add padding around vessel regions to include surrounding context"""
    # Convert to binary
    binary_mask = (mask > 127).astype(np.uint8)
    
    # Create structuring element for dilation (circular)
    struct = np.zeros((2*padding_radius+1, 2*padding_radius+1), dtype=np.uint8)
    y, x = np.ogrid[:2*padding_radius+1, :2*padding_radius+1]
    struct_mask = (x - padding_radius) ** 2 + (y - padding_radius) ** 2 <= padding_radius ** 2
    struct[struct_mask] = 1
    
    # Dilate to add padding around vessels
    padded_binary = binary_dilation(binary_mask, structure=struct)
    
    # Convert back to 0-255 range
    return (padded_binary * 255).astype(np.uint8)

def extract_vessel_patches(img_dir, mask_dir, output_img_dir, output_mask_dir, patch_size=64, min_vessel_ratio=0.05):
    """Extract 64x64 patches that contain vessels for meaningful inpainting comparison"""
    
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)
    
    mask_files = glob.glob(os.path.join(mask_dir, '*.png'))
    extracted_count = 0
    
    for mask_file in mask_files:
        filename = os.path.basename(mask_file)
        img_file = os.path.join(img_dir, filename)
        
        if not os.path.exists(img_file):
            continue
            
        # Load images
        img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        
        h, w = img.shape
        
        # Find patch location with maximum vessel content
        best_patch_info = None
        max_vessel_ratio = 0
        
        # Try multiple patch positions (grid search)
        step_size = patch_size // 4  # 25% overlap for better coverage
        
        for y in range(0, h - patch_size + 1, step_size):
            for x in range(0, w - patch_size + 1, step_size):
                # Extract potential patch
                mask_patch = mask[y:y+patch_size, x:x+patch_size]
                
                # Apply padding around vessels for better context
                padded_mask_patch = add_vessel_padding(mask_patch, padding_radius=3)
                
                # Calculate vessel ratio in this patch (using padded mask)
                vessel_pixels = np.sum(padded_mask_patch > 127)
                total_pixels = patch_size * patch_size
                vessel_ratio = vessel_pixels / total_pixels
                
                # Keep track of best patch with most vessels
                if vessel_ratio > max_vessel_ratio:
                    max_vessel_ratio = vessel_ratio
                    best_patch_info = (y, x, vessel_ratio)
        
        # Extract best patch if it has sufficient vessel content
        if best_patch_info and max_vessel_ratio >= min_vessel_ratio:
            y, x, ratio = best_patch_info
            
            img_patch = img[y:y+patch_size, x:x+patch_size]
            mask_patch = mask[y:y+patch_size, x:x+patch_size]
            
            # Add padding around vessels for better inpainting context
            padded_mask_patch = add_vessel_padding(mask_patch, padding_radius=3)
            
            # Save patches (using padded mask)
            cv2.imwrite(os.path.join(output_img_dir, filename), img_patch)
            cv2.imwrite(os.path.join(output_mask_dir, filename), padded_mask_patch)
            extracted_count += 1
            logger.debug(f"{filename}: vessel ratio {ratio:.3f} at position ({x},{y}) + 3px padding")
            
        else:
            # Fallback: extract center patch even if no vessels (for completeness)
            center_h, center_w = h // 2, w // 2
            start_h = center_h - patch_size // 2
            start_w = center_w - patch_size // 2
            
            img_patch = img[start_h:start_h+patch_size, start_w:start_w+patch_size]
            mask_patch = mask[start_h:start_h+patch_size, start_w:start_w+patch_size]
            
            # Add padding even for fallback patches
            padded_mask_patch = add_vessel_padding(mask_patch, padding_radius=3)
            
            cv2.imwrite(os.path.join(output_img_dir, filename), img_patch)
            cv2.imwrite(os.path.join(output_mask_dir, filename), padded_mask_patch)
            extracted_count += 1
            logger.debug(f"{filename}: no vessels found, using center patch + padding")

    logger.info(f"Extracted {extracted_count} vessel-focused patches ({patch_size}×{patch_size}) → {output_img_dir} | min_vessel_ratio={min_vessel_ratio:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract 64x64 patches for visualization")
    parser.add_argument("--img-dir", required=True, help="Input images directory")
    parser.add_argument("--mask-dir", required=True, help="Input masks directory") 
    parser.add_argument("--output-img", required=True, help="Output patches directory")
    parser.add_argument("--output-mask", required=True, help="Output mask patches directory")
    parser.add_argument("--patch-size", type=int, default=64, help="Patch size")
    
    parser.add_argument("--min-vessel-ratio", type=float, default=0.05, help="Minimum vessel ratio for patch selection")
    
    args = parser.parse_args()
    
    extract_vessel_patches(
        args.img_dir, args.mask_dir, 
        args.output_img, args.output_mask, 
        args.patch_size, args.min_vessel_ratio
    )