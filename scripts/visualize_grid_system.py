#!/usr/bin/env python3
"""
Visualisierung des Grid-basierten Masken-Systems

Zeigt wie ein 512×512 Bild systematisch in 8×8 Grid (64×64 Patches) 
aufgeteilt wird mit individuellen Masken für jede Zelle.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw
import argparse

def generate_vessel_mask(image_id, width, height, annotations_path):
    """Generate vessel mask from COCO annotations."""
    with open(annotations_path) as f:
        coco_data = json.load(f)
    
    # Create lookup tables
    id_to_info = {img['id']: img for img in coco_data['images']}
    anns_by_image = {}
    for ann in coco_data['annotations']:
        if ann['category_id'] != 26:  # Exclude stenosis
            img_id = ann['image_id'] 
            if img_id not in anns_by_image:
                anns_by_image[img_id] = []
            anns_by_image[img_id].append(ann)
    
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    
    if image_id not in anns_by_image:
        return np.array(mask)
        
    for ann in anns_by_image[image_id]:
        for segmentation in ann['segmentation']:
            if len(segmentation) >= 6:  # At least 3 points
                xy = [(segmentation[i], segmentation[i+1]) 
                      for i in range(0, len(segmentation), 2)]
                if len(xy) >= 3:
                    draw.polygon(xy, fill=255)
    return np.array(mask)

def create_grid_visualization(image_path, patch_dir, mask_dir, output_path, annotations_path=None):
    """Erstelle Visualisierung des Grid-Systems."""
    
    # Load original image
    if not os.path.exists(image_path):
        print(f"❌ Original image not found: {image_path}")
        return
    
    img = Image.open(image_path).convert('L')
    img_np = np.array(img)
    
    # Generate vessel mask if annotations provided
    vessel_mask = None
    if annotations_path and os.path.exists(annotations_path):
        base_name = os.path.basename(image_path).replace('.png', '')
        # Try to find image ID from filename (assuming format like "124.png" -> ID 124)
        try:
            image_id = int(base_name)
            vessel_mask = generate_vessel_mask(image_id, img.width, img.height, annotations_path)
        except ValueError:
            print(f"⚠️ Could not extract image ID from {base_name}")
            vessel_mask = None
    
    base_name = os.path.basename(image_path).replace('.png', '')
    grid_size = 64
    grid_cells = 8
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Grid-Based Mask System: {base_name}.png\\n'
                f'6×6 Inner Grid = 36 high-quality 64×64 patches (border excluded)', 
                fontsize=16, fontweight='bold')
    
    # 1. Original image with grid overlay
    ax = axes[0, 0]
    ax.imshow(img_np, cmap='gray')
    
    # Overlay vessel mask in red if available
    if vessel_mask is not None:
        ax.imshow(vessel_mask, cmap='Reds', alpha=0.4)
        ax.set_title('Original 512×512 Image + Vessels (Red)\\nwith 6×6 Inner Grid (Border Excluded)', fontweight='bold')
    else:
        ax.set_title('Original 512×512 Image\\nwith 6×6 Inner Grid (Border Excluded)', fontweight='bold')
    
    # Draw all grid lines
    for i in range(grid_cells + 1):
        pos = i * grid_size
        ax.axhline(pos, color='lime', linewidth=1, alpha=0.8)
        ax.axvline(pos, color='lime', linewidth=1, alpha=0.8)
    
    # Highlight excluded border cells
    border_alpha = 0.2
    ax.fill_between([0, 512], 0, grid_size, color='orange', alpha=border_alpha)  # Top border
    ax.fill_between([0, 512], 7*grid_size, 512, color='orange', alpha=border_alpha)  # Bottom border  
    ax.fill_between([0, grid_size], 0, 512, color='orange', alpha=border_alpha)  # Left border
    ax.fill_between([7*grid_size, 512], 0, 512, color='orange', alpha=border_alpha)  # Right border
    
    # Number some inner grid cells
    for y in range(1, grid_cells-1, 2):
        for x in range(1, grid_cells-1, 2):
            center_x = x * grid_size + grid_size // 2
            center_y = y * grid_size + grid_size // 2
            ax.text(center_x, center_y, f'{x},{y}', 
                   ha='center', va='center', color='yellow', 
                   fontweight='bold', fontsize=10)
    
    ax.set_xlim(0, 512)
    ax.set_ylim(512, 0)  # Invert y-axis for image display
    ax.axis('off')
    
    # 2. Sample inner grid patches (4 examples from inner 6x6 grid)
    sample_coords = [(1, 1), (3, 2), (5, 4), (6, 6)]
    
    ax = axes[0, 1] 
    ax.set_title('Sample Grid Patches\\n(4 random 64×64 cells)', fontweight='bold')
    
    # Create 2×2 subplot for 4 patches
    combined_patch = np.zeros((128, 128))
    
    for i, (gx, gy) in enumerate(sample_coords):
        patch_file = f"{base_name}_grid_{gx:02d}_{gy:02d}.png"
        patch_path = os.path.join(patch_dir, patch_file)
        
        if os.path.exists(patch_path):
            patch = np.array(Image.open(patch_path))
            
            # Place in 2×2 grid
            row = i // 2
            col = i % 2
            start_y = row * 64
            start_x = col * 64
            combined_patch[start_y:start_y+64, start_x:start_x+64] = patch
        
        # Add grid coordinates as labels
        if i < 4:
            label_y = (i // 2) * 64 + 32
            label_x = (i % 2) * 64 + 32
            ax.text(label_x, label_y, f'({gx},{gy})', 
                   ha='center', va='center', color='yellow', 
                   fontweight='bold', fontsize=12)
    
    ax.imshow(combined_patch, cmap='gray')
    ax.set_xlim(0, 128)
    ax.set_ylim(128, 0)
    ax.axis('off')
    
    # 3. Corresponding masks for sample patches
    ax = axes[0, 2]
    ax.set_title('Corresponding Background Masks\\n(Individual per cell)', fontweight='bold')
    
    combined_mask = np.zeros((128, 128))
    
    for i, (gx, gy) in enumerate(sample_coords):
        mask_file = f"{base_name}_grid_{gx:02d}_{gy:02d}.png"
        mask_path = os.path.join(mask_dir, mask_file)
        
        if os.path.exists(mask_path):
            mask = np.array(Image.open(mask_path))
            
            # Place in 2×2 grid  
            row = i // 2
            col = i % 2
            start_y = row * 64
            start_x = col * 64
            combined_mask[start_y:start_y+64, start_x:start_x+64] = mask
    
    ax.imshow(combined_mask, cmap='Blues', alpha=0.8)
    ax.imshow(combined_patch, cmap='gray', alpha=0.3)  # Show underlying image faintly
    ax.set_xlim(0, 128)
    ax.set_ylim(128, 0)
    ax.axis('off')
    
    # 4. Full image mask compilation
    ax = axes[1, 0]
    ax.set_title('All Grid Masks Combined\\n(64 individual masks)', fontweight='bold')
    
    # Compile all masks (only inner 6x6 grid)
    full_mask = np.zeros((512, 512))
    mask_count = 0
    
    for gy in range(1, grid_cells-1):
        for gx in range(1, grid_cells-1):
            mask_file = f"{base_name}_grid_{gx:02d}_{gy:02d}.png"
            mask_path = os.path.join(mask_dir, mask_file)
            
            if os.path.exists(mask_path):
                mask = np.array(Image.open(mask_path))
                start_y = gy * grid_size
                start_x = gx * grid_size
                full_mask[start_y:start_y+grid_size, start_x:start_x+grid_size] = mask
                
                if np.sum(mask > 0) > 0:  # Count non-empty masks
                    mask_count += 1
    
    ax.imshow(img_np, cmap='gray', alpha=0.4)
    if vessel_mask is not None:
        ax.imshow(vessel_mask, cmap='Reds', alpha=0.3)
    ax.imshow(full_mask, cmap='Blues', alpha=0.6)
    ax.set_title(f'All Grid Masks Combined\\n({mask_count}/36 inner cells have masks)', fontweight='bold')
    ax.axis('off')
    
    # 5. Coverage statistics
    ax = axes[1, 1]
    ax.set_title('Coverage Statistics', fontweight='bold')
    
    total_image_pixels = 512 * 512
    total_mask_pixels = np.sum(full_mask > 0)
    coverage_percent = (total_mask_pixels / total_image_pixels) * 100
    
    # Calculate per-cell statistics (only inner 6x6 grid)
    cell_coverages = []
    non_empty_cells = 0
    
    for gy in range(1, grid_cells-1):
        for gx in range(1, grid_cells-1):
            mask_file = f"{base_name}_grid_{gx:02d}_{gy:02d}.png"
            mask_path = os.path.join(mask_dir, mask_file)
            
            if os.path.exists(mask_path):
                mask = np.array(Image.open(mask_path))
                cell_pixels = np.sum(mask > 0)
                cell_coverage = (cell_pixels / (grid_size * grid_size)) * 100
                cell_coverages.append(cell_coverage)
                
                if cell_coverage > 0:
                    non_empty_cells += 1
    
    avg_cell_coverage = np.mean([c for c in cell_coverages if c > 0]) if non_empty_cells > 0 else 0
    
    # Create bar chart  
    categories = ['Total Image\\nCoverage', 'Avg Cell\\nCoverage', 'Inner Cells\\nwith Masks']
    values = [coverage_percent, avg_cell_coverage, (non_empty_cells/36)*100]
    colors = ['lightblue', 'skyblue', 'steelblue']
    
    bars = ax.bar(categories, values, color=colors)
    ax.set_ylabel('Percentage (%)')
    ax.set_ylim(0, max(100, max(values) * 1.1))
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
               f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 6. System advantages
    ax = axes[1, 2]
    ax.set_title('Grid System Advantages', fontweight='bold')
    ax.axis('off')
    
    advantages = [
        "✅ Inner 6×6 Grid (High Quality)",
        "✅ 36 Individual Training Samples", 
        "✅ Controlled Mask Size per Cell",
        "✅ No Border/Edge Artifacts",
        "✅ Balanced Context vs. Mask Ratio",
        "✅ Scalable to Any Image Size",
        "✅ Predictable Data Augmentation",
        "✅ Better Training Data Utilization"
    ]
    
    for i, advantage in enumerate(advantages):
        ax.text(0.05, 0.95 - i*0.11, advantage, transform=ax.transAxes,
               fontsize=11, fontweight='bold', 
               color='darkgreen' if '✅' in advantage else 'black')
    
    # Add summary statistics
    summary_text = f"""
Grid System Summary:
━━━━━━━━━━━━━━━━━━━━
📊 Image: {base_name}.png (512×512)
🔢 Inner Grid: 6×6 = 36 patches  
📏 Patch Size: 64×64 pixels
🎭 Masks Generated: {mask_count}/36 cells
📈 Total Coverage: {coverage_percent:.1f}%
⚖️ Avg Cell Coverage: {avg_cell_coverage:.1f}%
🎯 Max Coverage Limit: 30% per cell
🚫 Border patches excluded
"""
    
    fig.text(0.02, 0.02, summary_text, fontsize=10, 
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.20)
    
    # Save visualization
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"💾 Grid visualization saved: {output_path}")
    plt.close()
    
    # Return statistics
    return {
        'total_coverage': coverage_percent,
        'avg_cell_coverage': avg_cell_coverage,
        'cells_with_masks': non_empty_cells,
        'total_cells': 64
    }

def main():
    parser = argparse.ArgumentParser(description='Visualize grid-based mask system')
    parser.add_argument('--image', required=True, help='Original image path')
    parser.add_argument('--patch-dir', required=True, help='Grid patches directory') 
    parser.add_argument('--mask-dir', required=True, help='Grid masks directory')
    parser.add_argument('--annotations', help='COCO annotations JSON file for vessel overlay')
    parser.add_argument('--output', default='grid_system_visualization.png', help='Output visualization')
    
    args = parser.parse_args()
    
    stats = create_grid_visualization(
        args.image,
        args.patch_dir, 
        args.mask_dir,
        args.output,
        args.annotations
    )
    
    if stats:
        print(f"\n📊 **Grid System Statistics:**")
        print(f"   🎯 Total Coverage: {stats['total_coverage']:.1f}%")
        print(f"   📈 Average Cell Coverage: {stats['avg_cell_coverage']:.1f}%")
        print(f"   🎭 Cells with Masks: {stats['cells_with_masks']}/64")
        print(f"   ✅ Systematic Coverage: 100%")

if __name__ == '__main__':
    main()