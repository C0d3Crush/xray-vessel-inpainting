#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Precompute and cache vessel masks from COCO annotations.

This script generates binary masks once and saves them to disk, avoiding
repeated COCO parsing and mask rasterization during training.

Usage:
    python scripts/cache_masks.py \
        --annotations arcade/syntax/train/annotations/train.json \
        --images      arcade/syntax/train/images \
        --output      data/masks_cache/train

    # Then use in training:
    python train.py --train_mask data/masks_cache/train ...
"""

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm


STENOSIS_CATEGORY_ID = 26


def load_coco(ann_path):
    """Load COCO annotations and group by image."""
    with open(ann_path) as f:
        coco = json.load(f)

    id_to_info = {img['id']: img for img in coco['images']}
    anns_by_image = defaultdict(list)

    for ann in coco['annotations']:
        if ann['category_id'] != STENOSIS_CATEGORY_ID:
            anns_by_image[ann['image_id']].append(ann)

    return id_to_info, anns_by_image


def rasterize_mask(annotations, width, height):
    """Rasterize COCO polygon annotations into binary mask."""
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)

    for ann in annotations:
        for poly in ann['segmentation']:
            xy = list(zip(poly[0::2], poly[1::2]))
            if len(xy) >= 3:
                draw.polygon(xy, fill=255)

    return np.array(mask, dtype=np.uint8)


def generate_vessel_safe_mask(annotations, width, height, max_coverage=0.25, min_coverage=0.05):
    """Generate vessel-safe background masks using same logic as train.py."""
    # Step 1: Get vessel mask
    vessel_mask = rasterize_mask(annotations, width, height)
    
    # Step 2: Create vessel exclusion zone with safety margin
    safety_margin = 15
    kernel_size = max(5, safety_margin * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    vessel_exclusion = cv2.dilate(vessel_mask, kernel, iterations=2)
    
    # Step 3: Generate vessel-safe background mask
    bg_mask = np.zeros((height, width), dtype=np.uint8)
    total_pixels = width * height
    max_mask_pixels = int(total_pixels * max_coverage)
    min_mask_pixels = int(total_pixels * min_coverage)
    
    # Generate diverse shapes avoiding vessel regions
    shapes = ['circle', 'rectangle', 'ellipse', 'triangle', 'line', 'blob']
    successful_shapes = 0
    max_attempts = 150
    
    for attempt in range(max_attempts):
        current_pixels = np.sum(bg_mask > 0)
        
        if current_pixels >= max_mask_pixels:
            break
            
        shape_type = random.choice(shapes)
        temp_mask = generate_vessel_safe_shape(shape_type, width, height)
        
        if temp_mask is None:
            continue
            
        shape_pixels = np.sum(temp_mask > 0)
        if shape_pixels == 0:
            continue
        
        # Check vessel overlap - ZERO tolerance
        vessel_overlap = np.sum((temp_mask > 0) & (vessel_exclusion > 0))
        
        # Check existing mask overlap
        existing_overlap = np.sum((temp_mask > 0) & (bg_mask > 0))
        existing_ratio = existing_overlap / shape_pixels if shape_pixels > 0 else 1.0
        
        # Accept only shapes with NO vessel overlap and minimal existing overlap
        if vessel_overlap == 0 and existing_ratio < 0.15:
            bg_mask = np.maximum(bg_mask, temp_mask)
            successful_shapes += 1
    
    # Force minimum coverage if needed
    current_pixels = np.sum(bg_mask > 0)
    if current_pixels < min_mask_pixels:
        # Find completely free pixels
        free_coords = np.where((vessel_exclusion == 0) & (bg_mask == 0))
        
        if len(free_coords[0]) > 0:
            needed_pixels = min_mask_pixels - current_pixels
            available_pixels = len(free_coords[0])
            
            if available_pixels >= needed_pixels:
                # Randomly select pixels to meet minimum requirement
                indices = random.sample(range(len(free_coords[0])), min(needed_pixels, available_pixels))
                
                for idx in indices:
                    y, x = free_coords[0][idx], free_coords[1][idx]
                    cv2.circle(bg_mask, (x, y), 2, 255, -1)  # Small 2px circles
                
                # Final cleanup
                bg_mask[vessel_exclusion > 0] = 0
                successful_shapes += 1
    
    # Final vessel cleanup - ensure ZERO overlap
    bg_mask[vessel_exclusion > 0] = 0
    final_pixels = np.sum(bg_mask > 0)
    
    return bg_mask


def generate_vessel_safe_shape(shape_type, width, height):
    """Generate vessel-safe shapes using same logic as train.py."""
    temp_mask = np.zeros((height, width), dtype=np.uint8)
    
    try:
        if shape_type == 'circle':
            radius = random.randint(4, min(width, height) // 12)
            center_x = random.randint(radius, width - radius)
            center_y = random.randint(radius, height - radius)
            cv2.circle(temp_mask, (center_x, center_y), radius, 255, -1)
            
        elif shape_type == 'rectangle':
            w = random.randint(8, min(width, height) // 8)
            h = random.randint(8, min(width, height) // 8)
            x = random.randint(0, width - w)
            y = random.randint(0, height - h)
            cv2.rectangle(temp_mask, (x, y), (x + w, y + h), 255, -1)
            
        elif shape_type == 'ellipse':
            center_x = random.randint(width // 8, width - width // 8)
            center_y = random.randint(height // 8, height - height // 8)
            axes_x = random.randint(4, width // 12)
            axes_y = random.randint(4, height // 12)
            angle = random.randint(0, 180)
            cv2.ellipse(temp_mask, (center_x, center_y), (axes_x, axes_y), angle, 0, 360, 255, -1)
            
        elif shape_type == 'triangle':
            points = np.array([
                [random.randint(0, width), random.randint(0, height)],
                [random.randint(0, width), random.randint(0, height)],
                [random.randint(0, width), random.randint(0, height)]
            ], dtype=np.int32)
            cv2.fillPoly(temp_mask, [points], 255)
            
        elif shape_type == 'line':
            x1, y1 = random.randint(0, width), random.randint(0, height)
            x2, y2 = random.randint(0, width), random.randint(0, height)
            thickness = random.randint(3, 8)
            cv2.line(temp_mask, (x1, y1), (x2, y2), 255, thickness)
            
        elif shape_type == 'blob':
            # Irregular blob using multiple small circles
            num_circles = random.randint(3, 8)
            center_x = random.randint(width // 4, 3 * width // 4)
            center_y = random.randint(height // 4, 3 * height // 4)
            
            for _ in range(num_circles):
                offset_x = random.randint(-width // 12, width // 12)
                offset_y = random.randint(-height // 12, height // 12)
                radius = random.randint(2, width // 20)
                
                x = max(radius, min(width - radius, center_x + offset_x))
                y = max(radius, min(height - radius, center_y + offset_y))
                
                cv2.circle(temp_mask, (x, y), radius, 255, -1)
        
        return temp_mask
    
    except:
        return None


def main():
    parser = argparse.ArgumentParser(description="Precompute vessel masks from COCO annotations")
    parser.add_argument('--annotations', required=True, help='Path to COCO JSON')
    parser.add_argument('--images', required=True, help='Path to images directory')
    parser.add_argument('--output', required=True, help='Output directory for cached masks')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing masks')
    parser.add_argument('--vessel-safe-training', action='store_true', 
                        help='Generate vessel-safe background masks instead of vessel masks')
    args = parser.parse_args()

    # Load COCO
    print(f"Loading annotations from {args.annotations}...")
    id_to_info, anns_by_image = load_coco(args.annotations)
    print(f"  Found {len(id_to_info)} images")

    # Filter images that have annotations
    image_ids = [
        img_id for img_id in id_to_info
        if anns_by_image[img_id]
    ]
    print(f"  {len(image_ids)} images with vessel annotations")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Generate masks
    print(f"\nGenerating masks → {args.output}/")
    for img_id in tqdm(image_ids, desc="Processing"):
        info = id_to_info[img_id]
        filename = info['file_name']
        output_path = os.path.join(args.output, filename)

        # Skip if exists and not overwriting
        if os.path.exists(output_path) and not args.overwrite:
            continue

        # Generate mask based on mode
        W, H = info['width'], info['height']
        if getattr(args, 'vessel_safe_training', False):
            # Generate vessel-safe background masks
            mask = generate_vessel_safe_mask(anns_by_image[img_id], W, H)
        else:
            # Generate standard vessel masks
            mask = rasterize_mask(anns_by_image[img_id], W, H)

        # Save
        cv2.imwrite(output_path, mask)

    print(f"\n✓ Cached {len(image_ids)} masks to {args.output}/")
    print(f"\nUsage:")
    print(f"  python train.py --train_mask {args.output} ...")


if __name__ == '__main__':
    main()
