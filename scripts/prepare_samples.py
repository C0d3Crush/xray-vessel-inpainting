#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Populate samples/ with test images and their corresponding masks from ARCADE dataset.

This script:
1. Randomly selects N images from ARCADE dataset
2. Generates vessel masks from COCO annotations
3. Copies images to samples/test_img/
4. Saves masks to samples/test_mask/

Usage:
    python scripts/prepare_samples.py \
        --annotations arcade/syntax/val/annotations/val.json \
        --images      arcade/syntax/val/images \
        --num-samples 5

    # Use specific images
    python scripts/prepare_samples.py ... --image-ids 42,123,456

    # Overwrite existing samples
    python scripts/prepare_samples.py ... --overwrite
"""

import argparse
import json
import os
import random
import shutil
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

    # Filter images that have vessel annotations
    image_ids = [
        img_id for img_id in id_to_info
        if anns_by_image[img_id]
    ]

    return id_to_info, anns_by_image, image_ids


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


def main():
    parser = argparse.ArgumentParser(
        description="Populate samples/ with test images and masks from ARCADE"
    )
    parser.add_argument('--annotations', required=True,
                        help='Path to COCO JSON (e.g., arcade/syntax/val/annotations/val.json)')
    parser.add_argument('--images', required=True,
                        help='Path to images directory (e.g., arcade/syntax/val/images)')
    parser.add_argument('--num-samples', type=int, default=5,
                        help='Number of samples to extract (default: 5)')
    parser.add_argument('--image-ids', type=str, default=None,
                        help='Comma-separated image IDs to use (instead of random sampling)')
    parser.add_argument('--output-img', default='samples/test_img',
                        help='Output directory for images (default: samples/test_img)')
    parser.add_argument('--output-mask', default='samples/test_mask',
                        help='Output directory for masks (default: samples/test_mask)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing samples')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible sampling (default: 42)')
    args = parser.parse_args()

    # Load COCO
    print(f"Loading annotations from {args.annotations}...")
    id_to_info, anns_by_image, available_ids = load_coco(args.annotations)
    print(f"  Found {len(available_ids)} images with vessel annotations")

    # Select image IDs
    if args.image_ids:
        selected_ids = [int(x.strip()) for x in args.image_ids.split(',')]
        # Validate
        for img_id in selected_ids:
            if img_id not in id_to_info:
                print(f"Warning: Image ID {img_id} not found in dataset")
                selected_ids.remove(img_id)
        print(f"  Using {len(selected_ids)} specified images")
    else:
        random.seed(args.seed)
        selected_ids = random.sample(available_ids, min(args.num_samples, len(available_ids)))
        print(f"  Randomly selected {len(selected_ids)} images (seed={args.seed})")

    # Create output directories
    os.makedirs(args.output_img, exist_ok=True)
    os.makedirs(args.output_mask, exist_ok=True)

    # Check if directories not empty and no overwrite
    existing_imgs = list(Path(args.output_img).glob('*'))
    existing_masks = list(Path(args.output_mask).glob('*'))
    if (existing_imgs or existing_masks) and not args.overwrite:
        print(f"\nError: Output directories not empty. Use --overwrite to replace.")
        print(f"  {args.output_img}: {len(existing_imgs)} files")
        print(f"  {args.output_mask}: {len(existing_masks)} files")
        return

    # Clear directories if overwriting
    if args.overwrite:
        for f in existing_imgs:
            f.unlink()
        for f in existing_masks:
            f.unlink()
        print(f"  Cleared existing samples")

    # Process samples
    print(f"\nPreparing {len(selected_ids)} samples...")
    for img_id in tqdm(selected_ids, desc="Processing"):
        info = id_to_info[img_id]
        filename = info['file_name']
        W, H = info['width'], info['height']

        # Copy image
        src_img = os.path.join(args.images, filename)
        dst_img = os.path.join(args.output_img, filename)
        shutil.copy2(src_img, dst_img)

        # Generate mask
        mask = rasterize_mask(anns_by_image[img_id], W, H)
        dst_mask = os.path.join(args.output_mask, filename)
        cv2.imwrite(dst_mask, mask)

    print(f"\n✓ Prepared {len(selected_ids)} samples")
    print(f"  Images: {args.output_img}/")
    print(f"  Masks:  {args.output_mask}/")
    print(f"\nRun inference:")
    print(f"  python demo.py --ckpt checkpoints/best.pth --img_path {args.output_img} --mask_path {args.output_mask} --output_path samples/results")
    print(f"  make inference")


if __name__ == '__main__':
    main()
