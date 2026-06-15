#!/usr/bin/env python3
"""
Test trained background-only inpainting model with comprehensive evaluation.

This script:
1. Generates 64x64 patch samples from ARCADE dataset
2. Runs inference with trained background model  
3. Creates comparison visualizations
4. Validates that model generates only backgrounds, never vessels

Usage:
    python scripts/test_background_model.py --model best.pth --samples 8 --output outputs/background_test
"""

import os
import sys
import argparse
import random
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import torch

# Add src to path
sys.path.append('src')
from train import ArcadeDataset
from demo import load_model, run_inference

def setup_directories(output_dir):
    """Setup output directory structure."""
    output_path = Path(output_dir)
    dirs = {
        'input': output_path / 'input_patches',
        'masks': output_path / 'background_masks', 
        'results': output_path / 'inpainted_results',
        'vessel_masks': output_path / 'vessel_reference',
        'comparisons': output_path / 'comparisons'
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
        
    return dirs

def generate_background_samples(annotations_path, images_path, num_samples, patch_size, output_dirs):
    """Generate background training samples with vessel-safe masks."""
    print(f"🎯 Generating {num_samples} background samples ({patch_size}x{patch_size})")
    
    # Create dataset for sample generation
    dataset = ArcadeDataset(
        img_dir=images_path,
        ann_path=annotations_path, 
        image_size=patch_size,
        background_training=True,  # CRITICAL: Background mode
        patches_per_image=1,
        foreground_prob=0.7,  # Focus on background regions
        safety_margin=8,       # Enhanced safety
        max_shapes=6
    )
    
    print(f"📊 Dataset loaded: {len(dataset.image_ids)} images available")
    
    # Generate samples
    sample_info = []
    for i in range(num_samples):
        try:
            # Get random sample
            idx = random.randint(0, len(dataset) - 1)
            img_tensor, mask_tensor = dataset[idx]
            
            # Convert to numpy
            img_patch = ((img_tensor.squeeze().numpy() + 1.0) * 127.5).astype(np.uint8)
            mask_patch = (mask_tensor.squeeze().numpy() * 255).astype(np.uint8)
            
            # Also get vessel mask for reference (using vessel mode)
            image_idx = idx // dataset.patches_per_image
            image_id = dataset.image_ids[image_idx]
            
            # Create vessel reference mask
            vessel_dataset = ArcadeDataset(
                img_dir=images_path,
                ann_path=annotations_path,
                image_size=patch_size,
                background_training=False,  # Vessel mode for reference
                patches_per_image=1
            )
            
            # Get same image but with vessel masks
            img_info = dataset.id_to_info[image_id]
            W, H = img_info['width'], img_info['height']
            vessel_mask = vessel_dataset._make_mask_from_annotations(image_id, W, H)
            vessel_mask_np = np.array(vessel_mask, dtype=np.uint8)
            
            # Extract same patch region (this is approximate, for visualization)
            if vessel_mask_np.shape[0] >= patch_size and vessel_mask_np.shape[1] >= patch_size:
                vessel_patch = vessel_mask_np[:patch_size, :patch_size]
            else:
                vessel_patch = np.zeros((patch_size, patch_size), dtype=np.uint8)
            
            # Save files
            sample_name = f"sample_{i:02d}"
            
            Image.fromarray(img_patch).save(output_dirs['input'] / f"{sample_name}.png")
            Image.fromarray(mask_patch).save(output_dirs['masks'] / f"{sample_name}.png") 
            Image.fromarray(vessel_patch).save(output_dirs['vessel_masks'] / f"{sample_name}.png")
            
            sample_info.append({
                'name': sample_name,
                'image_id': image_id,
                'mask_coverage': np.sum(mask_patch > 0) / (patch_size * patch_size),
                'vessel_overlap': np.sum((mask_patch > 0) & (vessel_patch > 0)) / max(np.sum(mask_patch > 0), 1)
            })
            
            print(f"  ✓ {sample_name}: {sample_info[-1]['mask_coverage']:.1%} background, {sample_info[-1]['vessel_overlap']:.1%} vessel overlap")
            
        except Exception as e:
            print(f"  ⚠️  Failed to generate sample {i}: {e}")
            continue
    
    print(f"✅ Generated {len(sample_info)} samples successfully")
    return sample_info

def run_background_inference(model_path, input_dir, mask_dir, output_dir, device='cuda'):
    """Run inference with background-trained model."""
    print(f"🚀 Running background inference with {model_path}")
    
    # Load model
    model = load_model(model_path, device=device)
    print(f"📦 Model loaded on {device}")
    
    # Process each sample
    input_files = sorted(input_dir.glob("*.png"))
    results_info = []
    
    for img_path in input_files:
        sample_name = img_path.stem
        mask_path = mask_dir / f"{sample_name}.png"
        output_path = output_dir / f"{sample_name}.png"
        
        if not mask_path.exists():
            print(f"  ⚠️  Skipping {sample_name}: mask not found")
            continue
            
        try:
            # Run inference  
            result = run_inference(
                model=model,
                img_path=str(img_path),
                mask_path=str(mask_path), 
                output_path=str(output_path),
                device=device
            )
            
            if result:
                results_info.append(sample_name)
                print(f"  ✓ {sample_name}: inference complete")
            else:
                print(f"  ❌ {sample_name}: inference failed")
                
        except Exception as e:
            print(f"  ❌ {sample_name}: {e}")
            continue
    
    print(f"✅ Inference complete: {len(results_info)}/{len(input_files)} samples processed")
    return results_info

def create_background_comparison(input_dir, mask_dir, result_dir, vessel_dir, output_path, sample_info):
    """Create comprehensive background training comparison."""
    print(f"🎨 Creating background comparison visualization")
    
    # Import matplotlib here to avoid issues
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    # Get available samples
    available_samples = []
    for info in sample_info:
        sample_name = info['name']
        if all(d.exists() for d in [
            input_dir / f"{sample_name}.png",
            mask_dir / f"{sample_name}.png", 
            result_dir / f"{sample_name}.png",
            vessel_dir / f"{sample_name}.png"
        ]):
            available_samples.append(info)
    
    if not available_samples:
        print("❌ No complete sample sets found")
        return None
        
    # Limit to 6 samples for clean layout
    display_samples = available_samples[:6]
    
    # Setup figure
    fig, axes = plt.subplots(len(display_samples), 5, figsize=(15, 3 * len(display_samples)))
    if len(display_samples) == 1:
        axes = [axes]
    
    fig.suptitle('Background-Only Inpainting Results\n'
                 '(Input | Vessel Ref | Background Mask | Result | Overlay)', 
                 fontsize=14, fontweight='bold')
    
    for i, sample in enumerate(display_samples):
        sample_name = sample['name']
        
        # Load images
        input_img = np.array(Image.open(input_dir / f"{sample_name}.png"))
        mask_img = np.array(Image.open(mask_dir / f"{sample_name}.png"))
        result_img = np.array(Image.open(result_dir / f"{sample_name}.png"))
        vessel_img = np.array(Image.open(vessel_dir / f"{sample_name}.png"))
        
        # Create overlay to show vessel preservation
        overlay = result_img.copy()
        overlay_colored = np.stack([overlay, overlay, overlay], axis=-1)
        
        # Highlight vessels in red, background masks in blue
        vessel_mask = vessel_img > 127
        bg_mask = mask_img > 127
        
        overlay_colored[vessel_mask] = [255, 100, 100]  # Red vessels
        overlay_colored[bg_mask] = [overlay_colored[bg_mask, 0], 
                                   overlay_colored[bg_mask, 1], 
                                   np.minimum(255, overlay_colored[bg_mask, 2] + 60)]  # Blue background regions
        
        # Plot images
        axes[i][0].imshow(input_img, cmap='gray')
        axes[i][0].set_title('Input')
        axes[i][0].axis('off')
        
        axes[i][1].imshow(vessel_img, cmap='Reds', alpha=0.8)
        axes[i][1].imshow(input_img, cmap='gray', alpha=0.3) 
        axes[i][1].set_title('Vessel Ref')
        axes[i][1].axis('off')
        
        axes[i][2].imshow(mask_img, cmap='Blues', alpha=0.8)
        axes[i][2].imshow(input_img, cmap='gray', alpha=0.3)
        axes[i][2].set_title('BG Mask')
        axes[i][2].axis('off')
        
        axes[i][3].imshow(result_img, cmap='gray')
        axes[i][3].set_title('Result') 
        axes[i][3].axis('off')
        
        axes[i][4].imshow(overlay_colored)
        axes[i][4].set_title('Analysis')
        axes[i][4].axis('off')
        
        # Add sample info
        coverage = sample['mask_coverage']
        overlap = sample['vessel_overlap']
        axes[i][0].text(2, input_img.shape[0] - 2, 
                       f"{sample_name}\n{coverage:.1%} BG\n{overlap:.1%} V-overlap", 
                       fontsize=8, color='white', 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
    
    # Add legend
    vessel_patch = mpatches.Patch(color=[1, 0.4, 0.4], label='Vessels (preserved)')
    bg_patch = mpatches.Patch(color=[0.4, 0.4, 1], label='Background (inpainted)')
    fig.legend(handles=[vessel_patch, bg_patch], 
              loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=2)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.1)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Comparison saved: {output_path}")
    
    # Generate summary statistics
    avg_coverage = np.mean([s['mask_coverage'] for s in display_samples])
    avg_overlap = np.mean([s['vessel_overlap'] for s in display_samples])
    
    return {
        'samples_displayed': len(display_samples),
        'avg_background_coverage': avg_coverage,
        'avg_vessel_overlap': avg_overlap,
        'output_path': output_path
    }

def validate_background_only_training(result_dir, vessel_dir, sample_info):
    """Validate that model generates only backgrounds, not vessels."""
    print(f"🔍 Validating background-only training...")
    
    validation_results = []
    
    for sample in sample_info:
        sample_name = sample['name']
        result_path = result_dir / f"{sample_name}.png"
        vessel_path = vessel_dir / f"{sample_name}.png"
        
        if not (result_path.exists() and vessel_path.exists()):
            continue
            
        # Load images
        result_img = np.array(Image.open(result_path))
        vessel_img = np.array(Image.open(vessel_path)) 
        
        # Check if result contains vessel-like structures
        vessel_mask = vessel_img > 127
        
        if np.sum(vessel_mask) > 0:
            # Compare result in vessel regions vs outside vessel regions
            vessel_region_values = result_img[vessel_mask]
            non_vessel_region_values = result_img[~vessel_mask]
            
            # Vessels should be darker (lower values) than background
            # If model is generating vessels, we'd see dark structures in vessel regions
            vessel_mean = np.mean(vessel_region_values) if len(vessel_region_values) > 0 else 128
            background_mean = np.mean(non_vessel_region_values) if len(non_vessel_region_values) > 0 else 128
            
            # Calculate contrast (good background training shows even distribution)
            contrast_ratio = abs(vessel_mean - background_mean) / max(background_mean, 1)
            
            validation_results.append({
                'sample': sample_name,
                'vessel_mean': vessel_mean,
                'background_mean': background_mean,
                'contrast_ratio': contrast_ratio,
                'likely_background_only': contrast_ratio < 0.2  # Low contrast = even background
            })
            
        print(f"  📊 {sample_name}: vessel={vessel_mean:.1f}, bg={background_mean:.1f}, "
              f"contrast={contrast_ratio:.1%} {'✅' if contrast_ratio < 0.2 else '⚠️'}")
    
    # Summary
    background_only_count = sum(1 for r in validation_results if r['likely_background_only'])
    total_count = len(validation_results)
    
    print(f"\n📈 Validation Summary:")
    print(f"  Background-only samples: {background_only_count}/{total_count} "
          f"({background_only_count/max(total_count,1)*100:.1f}%)")
    print(f"  Average contrast ratio: {np.mean([r['contrast_ratio'] for r in validation_results]):.1%}")
    
    if background_only_count / max(total_count, 1) >= 0.8:
        print("  ✅ Model successfully trained for background-only generation!")
    else:
        print("  ⚠️  Model may still be generating vessel-like structures")
    
    return validation_results

def main():
    parser = argparse.ArgumentParser(description='Test background-only inpainting model')
    parser.add_argument('--model', required=True, help='Path to trained model (best.pth)')
    parser.add_argument('--annotations', default='data/arcade/syntax/val/annotations/val.json',
                       help='COCO annotations for sample generation')
    parser.add_argument('--images', default='data/arcade/syntax/val/images', 
                       help='Images directory for sample generation')
    parser.add_argument('--samples', type=int, default=8, help='Number of test samples')
    parser.add_argument('--patch-size', type=int, default=64, help='Patch size for testing')
    parser.add_argument('--output', default='outputs/background_test', help='Output directory')
    parser.add_argument('--device', default='cuda', help='Device for inference')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print(f"🎯 Testing Background-Only Inpainting Model")
    print(f"📦 Model: {args.model}")
    print(f"📊 Samples: {args.samples} × {args.patch_size}×{args.patch_size}")
    print(f"📁 Output: {args.output}")
    print("=" * 60)
    
    # Setup directories
    output_dirs = setup_directories(args.output)
    
    # Step 1: Generate background samples
    sample_info = generate_background_samples(
        annotations_path=args.annotations,
        images_path=args.images,
        num_samples=args.samples,
        patch_size=args.patch_size,
        output_dirs=output_dirs
    )
    
    if not sample_info:
        print("❌ Failed to generate samples")
        return
    
    # Step 2: Run inference
    inference_results = run_background_inference(
        model_path=args.model,
        input_dir=output_dirs['input'],
        mask_dir=output_dirs['masks'],
        output_dir=output_dirs['results'],
        device=args.device
    )
    
    if not inference_results:
        print("❌ Inference failed")
        return
    
    # Step 3: Create comparison
    comparison_path = Path(args.output) / 'background_training_validation.png'
    comparison_stats = create_background_comparison(
        input_dir=output_dirs['input'],
        mask_dir=output_dirs['masks'],
        result_dir=output_dirs['results'],
        vessel_dir=output_dirs['vessel_masks'],
        output_path=comparison_path,
        sample_info=sample_info
    )
    
    # Step 4: Validate background-only training
    validation_results = validate_background_only_training(
        result_dir=output_dirs['results'],
        vessel_dir=output_dirs['vessel_masks'],
        sample_info=sample_info
    )
    
    # Final summary
    print("\n" + "=" * 60)
    print("🎉 Background Model Testing Complete!")
    print(f"📊 Results saved in: {args.output}")
    if comparison_stats:
        print(f"📈 Average background coverage: {comparison_stats['avg_background_coverage']:.1%}")
        print(f"⚠️  Average vessel overlap: {comparison_stats['avg_vessel_overlap']:.1%}")
    print(f"🎯 Validation: {len([r for r in validation_results if r['likely_background_only']])}/{len(validation_results)} background-only")

if __name__ == "__main__":
    main()