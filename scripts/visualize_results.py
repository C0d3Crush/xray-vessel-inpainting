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


def create_comparison(img_path, mask_path, result_path):
    """Create side-by-side comparison: [Input | Mask | Result]"""
    # Read images
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    result = cv2.imread(str(result_path), cv2.IMREAD_GRAYSCALE)

    if img is None or mask is None or result is None:
        return None

    # Ensure all same size
    h, w = img.shape
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    result = cv2.resize(result, (w, h), interpolation=cv2.INTER_LINEAR)

    # Convert mask to 3-channel for red overlay
    mask_vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    mask_overlay = np.zeros_like(mask_vis)
    mask_overlay[:, :, 2] = mask  # Red channel for mask
    mask_vis = cv2.addWeighted(mask_vis, 0.7, mask_overlay, 0.3, 0)

    # Convert to BGR for consistency
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    result_bgr = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    color = (255, 255, 255)

    cv2.putText(img_bgr, 'Input', (10, 30), font, font_scale, color, thickness)
    cv2.putText(mask_vis, 'Mask', (10, 30), font, font_scale, color, thickness)
    cv2.putText(result_bgr, 'Result', (10, 30), font, font_scale, color, thickness)

    # Add separator lines
    separator = np.ones((h, 2, 3), dtype=np.uint8) * 255

    # Concatenate horizontally
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
        comparison = create_comparison(img_path, mask_path, result_path)
        if comparison is None:
            print(f"Warning: Failed to create comparison for {filename}")
            continue

        # Save
        output_file = output_path / filename
        cv2.imwrite(str(output_file), comparison)
        success_count += 1

    print(f"\n✓ Created {success_count} comparison images in {args.output}/")
    print(f"\nComparison format: [Input | Mask | Result]")


if __name__ == '__main__':
    main()
