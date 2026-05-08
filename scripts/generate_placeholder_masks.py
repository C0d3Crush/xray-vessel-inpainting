#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate simple placeholder masks for testing when ARCADE dataset is unavailable.

Creates basic rectangular/circular masks in the center of each image.
For production use, use prepare_samples.py with real ARCADE annotations.

Usage:
    python scripts/generate_placeholder_masks.py --input samples/test_img --output samples/test_mask
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def create_placeholder_mask(height, width, mask_type='center_rect'):
    """Create a simple placeholder mask for testing."""
    mask = np.zeros((height, width), dtype=np.uint8)

    if mask_type == 'center_rect':
        # Rectangle in center (40% of image)
        h_start = int(height * 0.3)
        h_end = int(height * 0.7)
        w_start = int(width * 0.3)
        w_end = int(width * 0.7)
        mask[h_start:h_end, w_start:w_end] = 255

    elif mask_type == 'center_circle':
        # Circle in center
        center = (width // 2, height // 2)
        radius = min(width, height) // 3
        cv2.circle(mask, center, radius, 255, -1)

    elif mask_type == 'random_strokes':
        # Random brush strokes
        num_strokes = np.random.randint(3, 8)
        for _ in range(num_strokes):
            x1, y1 = np.random.randint(0, width), np.random.randint(0, height)
            x2, y2 = np.random.randint(0, width), np.random.randint(0, height)
            thickness = np.random.randint(10, 30)
            cv2.line(mask, (x1, y1), (x2, y2), 255, thickness)

    return mask


def main():
    parser = argparse.ArgumentParser(description="Generate placeholder masks for testing")
    parser.add_argument('--input', default='samples/test_img',
                        help='Input images directory (default: samples/test_img)')
    parser.add_argument('--output', default='samples/test_mask',
                        help='Output masks directory (default: samples/test_mask)')
    parser.add_argument('--mask-type', default='random_strokes',
                        choices=['center_rect', 'center_circle', 'random_strokes'],
                        help='Type of mask to generate (default: random_strokes)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Get all images
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input directory {args.input} not found")
        return

    image_files = list(input_path.glob('*.png')) + list(input_path.glob('*.jpg'))
    image_files = [f for f in image_files if not f.name.startswith('.')]

    if not image_files:
        print(f"Error: No images found in {args.input}")
        return

    print(f"Found {len(image_files)} images")

    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate masks
    print(f"Generating {args.mask_type} masks...")
    for img_file in tqdm(image_files):
        # Read image to get dimensions
        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Could not read {img_file.name}")
            continue

        height, width = img.shape

        # Generate mask
        mask = create_placeholder_mask(height, width, args.mask_type)

        # Save mask
        output_file = output_path / img_file.name
        cv2.imwrite(str(output_file), mask)

    print(f"\n✓ Generated {len(image_files)} masks in {args.output}/")
    print(f"\nNote: These are placeholder masks for testing only!")
    print(f"For real masks, use: python scripts/prepare_samples.py with ARCADE dataset")
    print(f"\nRun inference:")
    print(f"  make inference")


if __name__ == '__main__':
    main()
