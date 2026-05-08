#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Preprocess COCO annotations into optimized pickle cache.

Parses COCO JSON once and saves as pickle for 10x faster dataset initialization.

Usage:
    python scripts/preprocess_coco.py \
        --annotations arcade/syntax/train/annotations/train.json \
        --output data/arcade_train.pkl

    python scripts/preprocess_coco.py \
        --annotations arcade/syntax/val/annotations/val.json \
        --output data/arcade_val.pkl
"""

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path


STENOSIS_CATEGORY_ID = 26


def preprocess_coco(ann_path):
    """Load and preprocess COCO annotations."""
    print(f"Loading {ann_path}...")
    with open(ann_path) as f:
        coco = json.load(f)

    # Build lookup dicts
    id_to_info = {img['id']: img for img in coco['images']}
    anns_by_image = defaultdict(list)

    for ann in coco['annotations']:
        if ann['category_id'] != STENOSIS_CATEGORY_ID:
            anns_by_image[ann['image_id']].append(ann)

    # Filter images with annotations
    image_ids = [
        img_id for img_id in id_to_info
        if anns_by_image[img_id]
    ]

    preprocessed = {
        'id_to_info': id_to_info,
        'anns_by_image': dict(anns_by_image),
        'image_ids': image_ids,
        'categories': coco.get('categories', []),
    }

    print(f"  Found {len(id_to_info)} images, {len(image_ids)} with annotations")
    return preprocessed


def main():
    parser = argparse.ArgumentParser(description="Preprocess COCO annotations to pickle")
    parser.add_argument('--annotations', required=True, help='Path to COCO JSON')
    parser.add_argument('--output', required=True, help='Output pickle path (e.g., data/arcade_train.pkl)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing pickle')
    args = parser.parse_args()

    output_path = Path(args.output)

    # Check if exists
    if output_path.exists() and not args.overwrite:
        print(f"Error: {output_path} exists. Use --overwrite to replace.")
        return

    # Preprocess
    data = preprocess_coco(args.annotations)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\n✓ Saved preprocessed annotations to {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1024:.1f} KB")
    print(f"\nUsage in train.py:")
    print(f"  Update ArcadeDataset.__init__() to load from pickle if available")


if __name__ == '__main__':
    main()
