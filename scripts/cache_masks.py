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


def main():
    parser = argparse.ArgumentParser(description="Precompute vessel masks from COCO annotations")
    parser.add_argument('--annotations', required=True, help='Path to COCO JSON')
    parser.add_argument('--images', required=True, help='Path to images directory')
    parser.add_argument('--output', required=True, help='Output directory for cached masks')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing masks')
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

        # Rasterize mask
        W, H = info['width'], info['height']
        mask = rasterize_mask(anns_by_image[img_id], W, H)

        # Save
        cv2.imwrite(output_path, mask)

    print(f"\n✓ Cached {len(image_ids)} masks to {args.output}/")
    print(f"\nUsage:")
    print(f"  python train.py --train_mask {args.output} ...")


if __name__ == '__main__':
    main()
