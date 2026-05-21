#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create side-by-side visualizations of input, mask, and inpainted result.

Usage:
    python scripts/visualize_results.py
    python scripts/visualize_results.py --input samples/test_img --mask samples/test_mask --result samples/results --output samples/comparisons
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def adaptive_mask_threshold(mask):
    """Determine optimal threshold for mask using multiple methods."""
    if mask.max() == mask.min():
        return mask.max() // 2  # Fallback for uniform images
    
    # Try Otsu thresholding first
    try:
        _, otsu_thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if otsu_thresh > 0:
            return otsu_thresh
    except:
        pass
    
    # Fallback to percentile-based threshold
    non_zero = mask[mask > 0]
    if len(non_zero) > 0:
        return np.percentile(non_zero, 50)  # Median of non-zero values
    
    # Ultimate fallback
    return 127


def find_optimal_vessel_region(mask, crop_size=16):
    """Find region with maximum vessel content using deterministic algorithm."""
    # Input validation
    if mask is None or mask.size == 0:
        raise ValueError("Invalid mask: empty or None")
    
    h, w = mask.shape
    if h < crop_size or w < crop_size:
        print(f"Warning: Image ({h}x{w}) smaller than crop size ({crop_size}x{crop_size}), using full image")
        return 0, 0, min(w, crop_size), min(h, crop_size)
    
    # Adaptive thresholding instead of arbitrary 127
    threshold = adaptive_mask_threshold(mask)
    mask_binary = (mask >= threshold).astype(np.uint8)
    
    vessel_pixels = np.sum(mask_binary)
    if vessel_pixels == 0:
        print(f"Warning: No vessel pixels found with threshold {threshold}, using center crop")
        x1 = w // 2 - crop_size // 2
        y1 = h // 2 - crop_size // 2
        return x1, y1, x1 + crop_size, y1 + crop_size
    
    print(f"Found {vessel_pixels} vessel pixels with adaptive threshold {threshold}")
    
    # Generate deterministic candidate positions using grid sampling
    stride = max(4, crop_size // 4)  # Sample every 4 pixels or crop_size/4
    candidates = []
    
    # Grid-based sampling for reproducible results
    for y in range(0, h - crop_size + 1, stride):
        for x in range(0, w - crop_size + 1, stride):
            candidates.append((x, y))
    
    # Add vessel centroid as high-priority candidate
    vessel_points = np.column_stack(np.where(mask_binary > 0))
    if len(vessel_points) > 0:
        cy, cx = np.mean(vessel_points, axis=0).astype(int)
        # Ensure centroid position is valid
        cx = max(crop_size//2, min(cx, w - crop_size//2))
        cy = max(crop_size//2, min(cy, h - crop_size//2))
        candidates.insert(0, (cx - crop_size//2, cy - crop_size//2))
    
    # Evaluate all candidates deterministically
    best_score = -1
    best_region = None
    
    for x, y in candidates:
        # Ensure valid crop coordinates
        x = max(0, min(x, w - crop_size))
        y = max(0, min(y, h - crop_size))
        
        # Extract crop region
        crop_mask = mask_binary[y:y+crop_size, x:x+crop_size]
        
        if crop_mask.shape != (crop_size, crop_size):
            continue  # Skip invalid crops
        
        # Score based on vessel density and spatial distribution
        vessel_count = np.sum(crop_mask)
        if vessel_count == 0:
            continue
        
        # Prefer regions with moderate density (not just dense clusters)
        density = vessel_count / (crop_size * crop_size)
        spatial_variance = np.var(crop_mask)  # Prefer structured patterns
        
        # Combined score: balance vessel content with spatial structure
        score = vessel_count * (1 + spatial_variance / 100)
        
        if score > best_score:
            best_score = score
            best_region = (x, y, x + crop_size, y + crop_size)
    
    if best_region is None:
        # Safe fallback with guaranteed valid coordinates
        x1 = max(0, min(w // 2 - crop_size // 2, w - crop_size))
        y1 = max(0, min(h // 2 - crop_size // 2, h - crop_size))
        best_region = (x1, y1, x1 + crop_size, y1 + crop_size)
        print(f"Using fallback region: ({x1}, {y1})")
    else:
        vessel_coverage = np.sum(mask_binary[best_region[1]:best_region[3], best_region[0]:best_region[2]])
        print(f"Selected region with {vessel_coverage} vessel pixels, score: {best_score:.2f}")
    
    return best_region


def validate_image(img_path, img_type="image"):
    """Validate image file exists and is readable."""
    if not img_path.exists():
        raise FileNotFoundError(f"{img_type} not found: {img_path}")
    
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read {img_type}: {img_path}")
    
    if img.size == 0:
        raise ValueError(f"Empty {img_type}: {img_path}")
    
    return img


def create_comparison(img_path, mask_path, result_path, zoom=True, target_size=64, crop_size=32):
    """Create side-by-side comparison: [Input | Mask | Result]"""
    try:
        # Validate and read all images with comprehensive error handling
        img = validate_image(img_path, "input image")
        mask = validate_image(mask_path, "mask image") 
        result = validate_image(result_path, "result image")
        
        # Validate image dimensions
        if len(img.shape) != 2:
            raise ValueError(f"Input image must be grayscale, got shape: {img.shape}")
        if len(mask.shape) != 2:
            raise ValueError(f"Mask image must be grayscale, got shape: {mask.shape}")
        if len(result.shape) != 2:
            raise ValueError(f"Result image must be grayscale, got shape: {result.shape}")
        
        # Validate target_size parameter
        if not isinstance(target_size, int) or target_size <= 0:
            raise ValueError(f"target_size must be positive integer, got: {target_size}")
            
        # Get reference dimensions from input image
        h, w = img.shape
        if h == 0 or w == 0:
            raise ValueError(f"Invalid input image dimensions: {h}x{w}")
        
        # Resize all images to same dimensions (input as reference)
        # Use EXACT same interpolation as training/inference for fair comparison
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        result = cv2.resize(result, (w, h), interpolation=cv2.INTER_LINEAR)
        
    except Exception as e:
        print(f"Error loading images: {e}")
        return None
    
    # Apply EXACT same normalization as training/inference for fair comparison
    # Input: already 0-255 grayscale
    # Result: should be 0-255 from demo.py output
    # Ensure both are uint8 without changing pixel values
    img = img.astype(np.uint8)
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    if zoom:
        # Find optimal region with robust vessel detection
        try:
            x1, y1, x2, y2 = find_optimal_vessel_region(mask, crop_size=crop_size)
            
            # Validate crop coordinates before applying
            if x2 - x1 != crop_size or y2 - y1 != crop_size:
                print(f"Warning: Invalid crop dimensions {x2-x1}x{y2-y1}, expected {crop_size}x{crop_size}")
                # Force safe center crop
                h, w = mask.shape
                x1 = max(0, min(w // 2 - crop_size // 2, w - crop_size))
                y1 = max(0, min(h // 2 - crop_size // 2, h - crop_size))
                x2, y2 = x1 + crop_size, y1 + crop_size
            
            # Apply validated crop to all images
            img = img[y1:y2, x1:x2]
            mask = mask[y1:y2, x1:x2]
            result = result[y1:y2, x1:x2]
            
            # Double-check final crop dimensions
            if img.shape != (crop_size, crop_size):
                print(f"Error: Final crop shape {img.shape}, expected ({crop_size}, {crop_size})")
                return None
                
        except Exception as e:
            print(f"Error during cropping: {e}, falling back to center crop")
            # Safe fallback
            h, w = mask.shape
            x1 = max(0, w // 2 - crop_size // 2)
            y1 = max(0, h // 2 - crop_size // 2)
            x2 = min(x1 + crop_size, w)
            y2 = min(y1 + crop_size, h)
            
            img = img[y1:y2, x1:x2]
            mask = mask[y1:y2, x1:x2]
            result = result[y1:y2, x1:x2]
            
    else:
        # If not zooming, resize full image to target size using training interpolation
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
        result = cv2.resize(result, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    # Convert mask to 3-channel for red overlay
    mask_vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Create strong red overlay where mask exists
    mask_binary = (mask > 0).astype(np.uint8) * 255  # Binary mask: 0 or 255
    red_overlay = np.zeros_like(mask_vis)
    red_overlay[:, :, 2] = mask_binary  # Strong red channel
    
    # Blend with stronger red visibility
    mask_vis = cv2.addWeighted(mask_vis, 0.5, red_overlay, 0.5, 0)

    # Convert to BGR for consistency
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    result_bgr = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    # Add labels (scale font based on image size)
    font = cv2.FONT_HERSHEY_SIMPLEX
    h_curr = img_bgr.shape[0]
    font_scale = max(0.5, h_curr / 400)  # Scale font with image size
    thickness = max(1, int(h_curr / 200))
    color = (255, 255, 255)
    
    # Add black outline for better readability
    outline_color = (0, 0, 0)
    outline_thickness = thickness + 1
    
    # Text position scaled with image
    text_y = max(30, int(h_curr * 0.08))
    text_x = max(10, int(img_bgr.shape[1] * 0.02))
    
    # Labels removed for cleaner comparison view

    # Add separator lines 
    sep_width = 1 if img_bgr.shape[0] <= 16 else max(2, img_bgr.shape[0] // 100)
    separator = np.ones((img_bgr.shape[0], sep_width, 3), dtype=np.uint8) * 255

    # Concatenate horizontally: Input | Mask | Result
    comparison = np.hstack([img_bgr, separator, mask_vis, separator, result_bgr])

    return comparison


def main():
    parser = argparse.ArgumentParser(
        description="Create side-by-side comparisons of input, mask, and result"
    )
    parser.add_argument('--input', default='samples/test_img',
                        help='Input images directory (default: samples/test_img)')
    parser.add_argument('--mask', default='samples/test_mask',
                        help='Mask images directory (default: samples/test_mask)')
    parser.add_argument('--result', default='samples/results',
                        help='Result images directory (default: samples/results)')
    parser.add_argument('--output', default='samples/comparisons',
                        help='Output directory for comparisons (default: samples/comparisons)')
    parser.add_argument('--no-zoom', action='store_true',
                        help='Disable zooming into mask regions (show full image)')
    parser.add_argument('--crop-size', type=int, default=32,
                        help='Size of cropped region when zooming (default: 32)')
    parser.add_argument('--size', type=int, default=64,
                        help='Target size for each comparison panel (default: 64)')
    args = parser.parse_args()

    # Check directories exist
    for dir_path, name in [(args.input, 'Input'), (args.mask, 'Mask'), (args.result, 'Result')]:
        if not Path(dir_path).exists():
            print(f"Error: {name} directory not found: {dir_path}")
            return

    # Get result files (these determine what we process)
    result_files = list(Path(args.result).glob('*.png'))
    if not result_files:
        print(f"Error: No result images found in {args.result}")
        return

    print(f"Found {len(result_files)} result images")

    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Process each result
    print("Creating comparisons...")
    success_count = 0
    for result_file in tqdm(result_files):
        filename = result_file.name
        img_path = Path(args.input) / filename
        mask_path = Path(args.mask) / filename
        result_path = result_file

        # Check all files exist
        if not img_path.exists():
            print(f"Warning: Input not found for {filename}")
            continue
        if not mask_path.exists():
            print(f"Warning: Mask not found for {filename}")
            continue

        # Create comparison
        zoom_enabled = not args.no_zoom
        comparison = create_comparison(img_path, mask_path, result_path, zoom=zoom_enabled, target_size=args.size, crop_size=args.crop_size)
        if comparison is None:
            print(f"Warning: Failed to create comparison for {filename}")
            continue

        # Save
        output_file = output_path / filename
        cv2.imwrite(str(output_file), comparison)
        success_count += 1

    print(f"\n✓ Created {success_count} comparison images in {args.output}/")
    print(f"\nComparison format: [Original | Mask | Inpainted]")
    if not args.no_zoom:
        print(f"{args.crop_size}x{args.crop_size} pixel crops (Total: {args.crop_size}x{args.crop_size*3 + 4})")
    else:
        print(f"Each panel size: {args.size}x{args.size} (Total: {args.size}x{args.size*3 + 4})")


if __name__ == '__main__':
    main()
