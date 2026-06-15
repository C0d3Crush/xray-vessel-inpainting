#!/usr/bin/env python3
"""
Patch-based inference script that extracts real 64x64 patches from full-resolution images
and runs inference on them without any resizing.

This creates true patch comparisons showing how the model performs on actual 64x64 regions.
"""

import argparse
import os
import json
import random
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from pathlib import Path
import sys

# Add src to path
sys.path.append('src')
from network.network_pro import Inpaint
from utils import load_checkpoint

def load_coco_annotations(ann_path):
    """Load COCO annotations and create lookup tables."""
    with open(ann_path) as f:
        coco = json.load(f)
    
    id_to_info = {img['id']: img for img in coco['images']}
    anns_by_image = {}
    
    for ann in coco['annotations']:
        if ann['category_id'] != 26:  # Exclude stenosis
            image_id = ann['image_id']
            if image_id not in anns_by_image:
                anns_by_image[image_id] = []
            anns_by_image[image_id].append(ann)
    
    return id_to_info, anns_by_image

def generate_vessel_mask(annotations, width, height, padding=0):
    """Generate vessel mask from COCO annotations with optional dilation padding."""
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)

    for ann in annotations:
        for segmentation in ann['segmentation']:
            if len(segmentation) >= 6:  # At least 3 points
                points = [(segmentation[i], segmentation[i+1]) for i in range(0, len(segmentation), 2)]
                draw.polygon(points, fill=255)

    mask_np = np.array(mask)
    if padding > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * padding + 1, 2 * padding + 1))
        mask_np = cv2.dilate(mask_np, kernel)
    return mask_np

def extract_patches_with_masks(image, vessel_mask, patch_size=64, num_patches=8, vessel_bias=0.7):
    """Extract random patches from image with vessel-biased sampling."""
    H, W = image.shape
    if H < patch_size or W < patch_size:
        return [], []
    
    patches = []
    masks = []
    
    # Get vessel pixels for biased sampling
    vessel_pixels = np.where(vessel_mask > 0)
    
    for _ in range(num_patches):
        if len(vessel_pixels[0]) > 0 and random.random() < vessel_bias:
            # Vessel-biased patch
            idx = random.randint(0, len(vessel_pixels[0]) - 1)
            center_y, center_x = vessel_pixels[0][idx], vessel_pixels[1][idx]
            
            # Random offset around vessel center
            offset_range = patch_size // 4
            offset_y = random.randint(-offset_range, offset_range)
            offset_x = random.randint(-offset_range, offset_range)
            
            y = center_y + offset_y - patch_size // 2
            x = center_x + offset_x - patch_size // 2
        else:
            # Random patch
            y = random.randint(0, H - patch_size)
            x = random.randint(0, W - patch_size)
        
        # Clamp to image boundaries
        y = max(0, min(y, H - patch_size))
        x = max(0, min(x, W - patch_size))
        
        # Extract patch
        img_patch = image[y:y+patch_size, x:x+patch_size]
        mask_patch = vessel_mask[y:y+patch_size, x:x+patch_size]
        
        # Only keep patches with some vessel content
        if np.sum(mask_patch > 0) > 10:  # At least 10 vessel pixels
            patches.append(img_patch)
            masks.append(mask_patch)
    
    return patches, masks

def run_patch_inference(model, patches, masks, device):
    """Run inference on patch arrays."""
    results = []
    
    for img_patch, mask_patch in zip(patches, masks):
        # Convert to tensors and normalize
        img_tensor = torch.from_numpy((img_patch.astype(np.float32) / 255.0) * 2.0 - 1.0)
        mask_tensor = torch.from_numpy(mask_patch.astype(np.float32) / 255.0)
        
        # Add batch and channel dimensions
        img_tensor = img_tensor[None, None, :, :].to(device)
        mask_tensor = mask_tensor[None, None, :, :].to(device)
        
        # Run inference
        with torch.no_grad():
            result = model(img_tensor, mask_tensor)
        
        # Convert back to numpy
        result = torch.clip(result, -1.0, 1.0) * 0.5 + 0.5
        result = result[0, 0].cpu().numpy() * 255.0
        
        results.append(result.astype(np.uint8))
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Extract real patches and run inference")
    parser.add_argument('--ckpt', required=True, help='Model checkpoint path')
    parser.add_argument('--annotations', required=True, help='COCO annotations JSON')
    parser.add_argument('--images', required=True, help='Images directory')
    parser.add_argument('--output-dir', default='outputs/patches', help='Output directory')
    parser.add_argument('--num-images', type=int, default=4, help='Number of images to process')
    parser.add_argument('--patches-per-image', type=int, default=2, help='Patches per image')
    parser.add_argument('--patch-size', type=int, default=64, help='Patch size')
    parser.add_argument('--vessel-bias', type=float, default=0.7, help='Vessel sampling bias')
    parser.add_argument('--device', default='cpu', help='Device to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--mask-padding', type=int, default=0, help='Dilation radius in pixels added around vessel annotations')
    
    args = parser.parse_args()
    
    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load model
    device = torch.device(args.device)
    model = Inpaint(input_size=args.patch_size).to(device)
    model = load_checkpoint(args.ckpt, model, device)
    model.eval()
    
    # Load annotations
    print(f"Loading annotations from {args.annotations}...")
    id_to_info, anns_by_image = load_coco_annotations(args.annotations)
    available_ids = list(anns_by_image.keys())
    print(f"Found {len(available_ids)} images with vessel annotations")
    
    # Select random images
    selected_ids = random.sample(available_ids, min(args.num_images, len(available_ids)))
    print(f"Processing {len(selected_ids)} images...")
    
    # Create output directories
    output_base = Path(args.output_dir)
    (output_base / 'original').mkdir(parents=True, exist_ok=True)
    (output_base / 'mask').mkdir(parents=True, exist_ok=True)
    (output_base / 'result').mkdir(parents=True, exist_ok=True)
    
    patch_count = 0
    
    for img_id in selected_ids:
        info = id_to_info[img_id]
        img_path = os.path.join(args.images, info['file_name'])
        
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            continue
        
        # Load image
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Warning: Could not load image: {img_path}")
            continue
        
        # Generate vessel mask
        vessel_mask = generate_vessel_mask(anns_by_image[img_id], info['width'], info['height'], padding=args.mask_padding)
        
        # Extract patches
        patches, masks = extract_patches_with_masks(
            image, vessel_mask, args.patch_size, args.patches_per_image, args.vessel_bias
        )
        
        if not patches:
            print(f"Warning: No suitable patches found in image {img_id}")
            continue
        
        # Run inference
        results = run_patch_inference(model, patches, masks, device)
        
        # Save patches
        for i, (patch, mask, result) in enumerate(zip(patches, masks, results)):
            filename = f"img_{img_id}_patch_{i:02d}.png"
            
            cv2.imwrite(str(output_base / 'original' / filename), patch)
            cv2.imwrite(str(output_base / 'mask' / filename), mask)
            cv2.imwrite(str(output_base / 'result' / filename), result)
            
            patch_count += 1
    
    print(f"✓ Generated {patch_count} patch comparisons")
    print(f"  Original patches: {output_base / 'original'}/")
    print(f"  Vessel masks: {output_base / 'mask'}/") 
    print(f"  Inpainted results: {output_base / 'result'}/")
    print(f"\nCreate comparison:")
    print(f"  python scripts/create_training_comparison.py --patch-img {output_base / 'original'} --patch-mask {output_base / 'mask'} --patch-result {output_base / 'result'} --output {output_base / 'patch_comparison.png'} --title 'Real 64x64 Patch Inference Results'")

if __name__ == '__main__':
    main()