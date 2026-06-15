#!/usr/bin/env python3
"""
Comprehensive Mask & Vessel Overview Visualization

Creates detailed grid-based overview showing:
- Original X-ray images 
- Vessel structures (red)
- Background training masks (blue)
- 8×8 patch boundaries
- Overlap analysis per patch
- Coverage statistics

Zeigt eine komplette Übersicht aller Masken und Gefäße in einem strukturierten Layout.
"""

import os
import sys
import json
import random
import argparse
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.append('src')

class MaskOverviewGenerator:
    def __init__(self, annotations_path, images_path, grid_size=64):
        self.annotations_path = annotations_path
        self.images_path = images_path
        self.grid_size = grid_size
        
        # Load COCO annotations
        with open(annotations_path) as f:
            self.coco_data = json.load(f)
            
        # Create lookup tables
        self.id_to_info = {img['id']: img for img in self.coco_data['images']}
        self.anns_by_image = {}
        for ann in self.coco_data['annotations']:
            if ann['category_id'] != 26:  # Exclude stenosis
                image_id = ann['image_id'] 
                if image_id not in self.anns_by_image:
                    self.anns_by_image[image_id] = []
                self.anns_by_image[image_id].append(ann)
                
        # Filter to only images with annotations
        self.image_ids = [img_id for img_id in self.id_to_info 
                         if img_id in self.anns_by_image]
                         
        print(f"📁 Loaded {len(self.image_ids)} annotated images")

    def generate_vessel_mask(self, image_id, width, height):
        """Generate vessel mask from COCO annotations."""
        mask = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(mask)
        
        if image_id not in self.anns_by_image:
            return mask
            
        for ann in self.anns_by_image[image_id]:
            for segmentation in ann['segmentation']:
                if len(segmentation) >= 6:  # At least 3 points
                    xy = [(segmentation[i], segmentation[i+1]) 
                          for i in range(0, len(segmentation), 2)]
                    if len(xy) >= 3:
                        draw.polygon(xy, fill=255)
        return mask

    def generate_background_mask(self, vessel_mask, width, height, safety_margin=15, 
                                num_shapes=8, max_coverage=0.25):
        """Generate safe background mask avoiding vessels."""
        vessel_np = np.array(vessel_mask, dtype=np.uint8)
        
        # Create vessel exclusion zone with enhanced safety margin
        kernel_size = max(5, safety_margin * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        vessel_exclusion = cv2.dilate(vessel_np, kernel, iterations=2)
        
        # Generate random background patches
        bg_mask = np.zeros((height, width), dtype=np.uint8)
        successful_shapes = 0
        max_attempts = 150
        total_pixels = width * height
        max_mask_pixels = int(total_pixels * max_coverage)
        
        for shape_idx in range(num_shapes):
            current_mask_pixels = np.sum(bg_mask > 0)
            if current_mask_pixels >= max_mask_pixels:
                break
                
            for attempt in range(max_attempts):
                shape_type = random.choice(['circle', 'rectangle', 'ellipse', 'blob'])
                temp_mask = self._generate_shape(shape_type, width, height)
                
                if temp_mask is None:
                    continue
                
                # Check vessel overlap
                shape_pixels = np.sum(temp_mask > 0)
                if shape_pixels == 0:
                    continue
                    
                vessel_overlap = np.sum((temp_mask > 0) & (vessel_exclusion > 0))
                vessel_overlap_ratio = vessel_overlap / shape_pixels
                
                # Check existing mask overlap
                existing_overlap = np.sum((temp_mask > 0) & (bg_mask > 0))
                existing_overlap_ratio = existing_overlap / shape_pixels
                
                # Check coverage limit
                combined_mask = np.maximum(bg_mask, temp_mask)
                new_total_pixels = np.sum(combined_mask > 0)
                
                # Accept shape with strict vessel avoidance
                if (vessel_overlap_ratio < 0.05 and  # Very low vessel overlap
                    existing_overlap_ratio < 0.15 and  # Some mask overlap OK
                    new_total_pixels <= max_mask_pixels):
                    
                    bg_mask = combined_mask
                    successful_shapes += 1
                    break
        
        return Image.fromarray(bg_mask, mode='L'), successful_shapes, vessel_exclusion

    def _generate_shape(self, shape_type, width, height):
        """Generate a single random shape."""
        temp_mask = np.zeros((height, width), dtype=np.uint8)
        
        if shape_type == 'circle':
            radius = random.randint(8, min(width, height) // 8)
            center_x = random.randint(radius, width - radius)
            center_y = random.randint(radius, height - radius)
            cv2.circle(temp_mask, (center_x, center_y), radius, 255, -1)
            
        elif shape_type == 'rectangle':
            w = random.randint(12, min(width, height) // 6)
            h = random.randint(12, min(width, height) // 6)
            x = random.randint(0, width - w)
            y = random.randint(0, height - h)
            cv2.rectangle(temp_mask, (x, y), (x + w, y + h), 255, -1)
            
        elif shape_type == 'ellipse':
            center_x = random.randint(width // 6, 5 * width // 6)
            center_y = random.randint(height // 6, 5 * height // 6)
            axes_x = random.randint(8, min(width, height) // 10)
            axes_y = random.randint(8, min(width, height) // 10)
            angle = random.randint(0, 180)
            cv2.ellipse(temp_mask, (center_x, center_y), (axes_x, axes_y), 
                       angle, 0, 360, 255, -1)
                       
        elif shape_type == 'blob':
            center_x = random.randint(width // 6, 5 * width // 6)
            center_y = random.randint(height // 6, 5 * height // 6)
            base_size = random.randint(8, min(width, height) // 12)
            
            # Create irregular blob
            num_points = random.randint(6, 10)
            angles = np.linspace(0, 2*np.pi, num_points+1)[:-1]
            points = []
            for angle in angles:
                radius = base_size * random.uniform(0.6, 1.4)
                x = int(center_x + radius * np.cos(angle))
                y = int(center_y + radius * np.sin(angle))
                points.append([x, y])
            
            points = np.array(points, dtype=np.int32)
            cv2.fillPoly(temp_mask, [points], 255)
        
        return temp_mask

    def analyze_grid_overlaps(self, vessel_mask, bg_mask, width, height):
        """Analyze vessel/background overlaps per 8×8 grid cell."""
        vessel_np = np.array(vessel_mask, dtype=np.uint8)
        bg_np = np.array(bg_mask, dtype=np.uint8)
        
        grid_cells = width // self.grid_size
        overlap_data = []
        
        for grid_y in range(grid_cells):
            for grid_x in range(grid_cells):
                # Extract cell region
                start_x = grid_x * self.grid_size
                start_y = grid_y * self.grid_size
                end_x = start_x + self.grid_size
                end_y = start_y + self.grid_size
                
                cell_vessel = vessel_np[start_y:end_y, start_x:end_x]
                cell_bg = bg_np[start_y:end_y, start_x:end_x]
                
                # Calculate statistics
                vessel_pixels = np.sum(cell_vessel > 0)
                bg_pixels = np.sum(cell_bg > 0)
                overlap_pixels = np.sum((cell_vessel > 0) & (cell_bg > 0))
                total_pixels = self.grid_size * self.grid_size
                
                vessel_coverage = vessel_pixels / total_pixels
                bg_coverage = bg_pixels / total_pixels
                overlap_ratio = overlap_pixels / bg_pixels if bg_pixels > 0 else 0.0
                
                overlap_data.append({
                    'grid_x': grid_x,
                    'grid_y': grid_y,
                    'vessel_coverage': vessel_coverage,
                    'bg_coverage': bg_coverage,
                    'overlap_ratio': overlap_ratio,
                    'vessel_pixels': vessel_pixels,
                    'bg_pixels': bg_pixels,
                    'overlap_pixels': overlap_pixels
                })
        
        return overlap_data

    def create_comprehensive_overview(self, image_id, output_path=None):
        """Create comprehensive overview visualization."""
        # Load image
        img_info = self.id_to_info[image_id]
        img_path = os.path.join(self.images_path, img_info['file_name'])
        
        if not os.path.exists(img_path):
            print(f"❌ Image not found: {img_path}")
            return None
            
        img = Image.open(img_path).convert('L')
        width, height = img.size
        
        if width != 512 or height != 512:
            print(f"⚠️ Image size {width}×{height} is not 512×512")
            return None
        
        print(f"🖼️  Processing {img_info['file_name']} ({width}×{height})")
        
        # Generate masks
        vessel_mask = self.generate_vessel_mask(image_id, width, height)
        bg_mask, num_shapes, vessel_exclusion = self.generate_background_mask(
            vessel_mask, width, height
        )
        
        print(f"🎭 Generated background mask with {num_shapes} shapes")
        
        # Analyze grid overlaps
        overlap_data = self.analyze_grid_overlaps(vessel_mask, bg_mask, width, height)
        
        # Create visualization
        fig = plt.figure(figsize=(20, 16))
        gs = gridspec.GridSpec(3, 4, height_ratios=[2, 2, 1], width_ratios=[1, 1, 1, 1])
        
        # Convert to RGB for overlay
        img_rgb = np.stack([np.array(img)] * 3, axis=-1)
        vessel_rgb = np.array(vessel_mask)
        bg_rgb = np.array(bg_mask)
        exclusion_rgb = vessel_exclusion
        
        # 1. Original Image with Grid
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(img_rgb, cmap='gray')
        self._draw_grid_lines(ax1, width, height)
        ax1.set_title('Original X-ray + 8×8 Grid', fontweight='bold', fontsize=12)
        ax1.axis('off')
        
        # 2. Vessel Mask
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(img_rgb, cmap='gray', alpha=0.7)
        ax2.imshow(vessel_rgb, alpha=0.6, cmap='Reds')
        self._draw_grid_lines(ax2, width, height)
        ax2.set_title('Vessel Structures (Red)', fontweight='bold', fontsize=12, color='red')
        ax2.axis('off')
        
        # 3. Background Masks
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(img_rgb, cmap='gray', alpha=0.7)
        ax3.imshow(bg_rgb, alpha=0.6, cmap='Blues')
        self._draw_grid_lines(ax3, width, height)
        ax3.set_title('Background Training Masks (Blue)', fontweight='bold', fontsize=12, color='blue')
        ax3.axis('off')
        
        # 4. Combined Overlay
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.imshow(img_rgb, cmap='gray', alpha=0.7)
        ax4.imshow(vessel_rgb, alpha=0.4, cmap='Reds')
        ax4.imshow(bg_rgb, alpha=0.4, cmap='Blues')
        self._draw_grid_lines(ax4, width, height)
        ax4.set_title('Combined: Vessels + Masks', fontweight='bold', fontsize=12)
        ax4.axis('off')
        
        # 5. Vessel Exclusion Zone
        ax5 = fig.add_subplot(gs[1, 0])
        ax5.imshow(img_rgb, cmap='gray', alpha=0.7)
        ax5.imshow(vessel_rgb, alpha=0.6, cmap='Reds')
        ax5.imshow(exclusion_rgb, alpha=0.3, cmap='Oranges')
        self._draw_grid_lines(ax5, width, height)
        ax5.set_title('Safety Exclusion Zone (Orange)', fontweight='bold', fontsize=12, color='orange')
        ax5.axis('off')
        
        # 6. Overlap Analysis Heatmap
        ax6 = fig.add_subplot(gs[1, 1])
        overlap_grid = self._create_overlap_heatmap(overlap_data, width // self.grid_size)
        im = ax6.imshow(overlap_grid, cmap='RdYlGn_r', vmin=0, vmax=0.2)
        ax6.set_title('Vessel-Mask Overlap % per Cell', fontweight='bold', fontsize=12)
        plt.colorbar(im, ax=ax6, fraction=0.046, pad=0.04, label='Overlap %')
        ax6.set_xticks(range(8))
        ax6.set_yticks(range(8))
        
        # 7. Coverage Analysis Heatmap
        ax7 = fig.add_subplot(gs[1, 2])
        coverage_grid = self._create_coverage_heatmap(overlap_data, width // self.grid_size)
        im2 = ax7.imshow(coverage_grid, cmap='Blues', vmin=0, vmax=0.3)
        ax7.set_title('Background Mask Coverage % per Cell', fontweight='bold', fontsize=12)
        plt.colorbar(im2, ax=ax7, fraction=0.046, pad=0.04, label='Coverage %')
        ax7.set_xticks(range(8))
        ax7.set_yticks(range(8))
        
        # 8. Quality Assessment
        ax8 = fig.add_subplot(gs[1, 3])
        quality_grid = self._create_quality_heatmap(overlap_data, width // self.grid_size)
        im3 = ax8.imshow(quality_grid, cmap='RdYlGn', vmin=0, vmax=1)
        ax8.set_title('Patch Quality Score (Green=Good)', fontweight='bold', fontsize=12)
        plt.colorbar(im3, ax=ax8, fraction=0.046, pad=0.04, label='Quality Score')
        ax8.set_xticks(range(8))
        ax8.set_yticks(range(8))
        
        # 9. Statistics Summary
        ax9 = fig.add_subplot(gs[2, :])
        ax9.axis('off')
        
        # Calculate summary statistics
        total_vessel_pixels = np.sum(vessel_rgb > 0)
        total_bg_pixels = np.sum(bg_rgb > 0)
        total_overlap_pixels = np.sum((vessel_rgb > 0) & (bg_rgb > 0))
        total_pixels = width * height
        
        vessel_coverage = total_vessel_pixels / total_pixels
        bg_coverage = total_bg_pixels / total_pixels
        overall_overlap = total_overlap_pixels / total_bg_pixels if total_bg_pixels > 0 else 0
        
        # Count quality patches
        good_patches = sum(1 for data in overlap_data if data['overlap_ratio'] < 0.05)
        patches_with_masks = sum(1 for data in overlap_data if data['bg_pixels'] > 0)
        
        avg_patch_overlap = np.mean([data['overlap_ratio'] for data in overlap_data])
        max_patch_overlap = np.max([data['overlap_ratio'] for data in overlap_data])
        
        stats_text = f"""
🖼️ **Image:** {img_info['file_name']} | 📐 **Size:** {width}×{height} | 🎯 **Grid:** 8×8 ({self.grid_size}×{self.grid_size} cells)

📊 **Coverage Statistics:**
   • 🩸 Vessel Coverage: {vessel_coverage:.1%} ({total_vessel_pixels:,} pixels)
   • 🔵 Background Mask Coverage: {bg_coverage:.1%} ({total_bg_pixels:,} pixels) 
   • ⚠️ Overall Overlap: {overall_overlap:.1%} ({total_overlap_pixels:,} pixels)

🎯 **Patch Quality Analysis:**
   • ✅ Good Patches (< 5% overlap): {good_patches}/64 ({good_patches/64:.1%})
   • 🎭 Patches with Masks: {patches_with_masks}/64 ({patches_with_masks/64:.1%})
   • 📊 Average Patch Overlap: {avg_patch_overlap:.1%}
   • 🚨 Maximum Patch Overlap: {max_patch_overlap:.1%}

🎨 **Shape Generation:**
   • 🔵 Background Shapes Created: {num_shapes}
   • 🛡️ Safety Margin: 15px vessel exclusion
   • 🎲 Shape Types: Circle, Rectangle, Ellipse, Blob
        """
        
        ax9.text(0.02, 0.98, stats_text, transform=ax9.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.8", facecolor="lightgray", alpha=0.8))
        
        plt.suptitle(f'Comprehensive Mask & Vessel Overview: {img_info["file_name"]}', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.94, bottom=0.02)
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"💾 Saved overview: {output_path}")
        else:
            plt.show()
            
        plt.close()
        
        return {
            'vessel_coverage': vessel_coverage,
            'bg_coverage': bg_coverage,
            'overall_overlap': overall_overlap,
            'good_patches': good_patches,
            'patches_with_masks': patches_with_masks,
            'overlap_data': overlap_data
        }

    def _draw_grid_lines(self, ax, width, height):
        """Draw 8×8 grid lines on the plot."""
        grid_cells = width // self.grid_size
        
        # Vertical lines
        for i in range(1, grid_cells):
            x = i * self.grid_size
            ax.axvline(x=x, color='white', linewidth=1, alpha=0.7)
            
        # Horizontal lines  
        for i in range(1, grid_cells):
            y = i * self.grid_size
            ax.axhline(y=y, color='white', linewidth=1, alpha=0.7)

    def _create_overlap_heatmap(self, overlap_data, grid_size):
        """Create overlap heatmap grid."""
        grid = np.zeros((grid_size, grid_size))
        for data in overlap_data:
            grid[data['grid_y'], data['grid_x']] = data['overlap_ratio']
        return grid

    def _create_coverage_heatmap(self, overlap_data, grid_size):
        """Create coverage heatmap grid."""
        grid = np.zeros((grid_size, grid_size))
        for data in overlap_data:
            grid[data['grid_y'], data['grid_x']] = data['bg_coverage']
        return grid

    def _create_quality_heatmap(self, overlap_data, grid_size):
        """Create quality score heatmap."""
        grid = np.zeros((grid_size, grid_size))
        for data in overlap_data:
            # Quality score: higher is better
            # Good mask coverage with low vessel overlap = high quality
            mask_score = min(data['bg_coverage'] * 4, 1.0)  # Normalize coverage
            overlap_penalty = data['overlap_ratio'] * 5  # Penalize overlap
            quality = max(0, mask_score - overlap_penalty)
            grid[data['grid_y'], data['grid_x']] = quality
        return grid

    def create_batch_overview(self, output_dir, num_images=5):
        """Create overview for multiple images."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Select random images
        selected_ids = random.sample(self.image_ids, min(num_images, len(self.image_ids)))
        
        print(f"🎯 Creating batch overview for {len(selected_ids)} images")
        
        all_results = []
        
        for i, image_id in enumerate(tqdm(selected_ids, desc="Generating overviews")):
            img_info = self.id_to_info[image_id]
            base_name = img_info['file_name'].replace('.png', '')
            output_path = os.path.join(output_dir, f"{base_name}_mask_overview.png")
            
            result = self.create_comprehensive_overview(image_id, output_path)
            if result:
                result['image_name'] = img_info['file_name']
                all_results.append(result)
        
        # Create summary report
        self._create_summary_report(all_results, output_dir)
        
        print(f"✅ Batch overview complete!")
        print(f"📁 Overview images: {output_dir}/")
        print(f"📊 Summary report: {output_dir}/summary_report.txt")

    def _create_summary_report(self, results, output_dir):
        """Create text summary report."""
        if not results:
            return
            
        report_path = os.path.join(output_dir, "summary_report.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("MASK & VESSEL OVERVIEW SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Overall statistics
            avg_vessel = np.mean([r['vessel_coverage'] for r in results])
            avg_bg = np.mean([r['bg_coverage'] for r in results])
            avg_overlap = np.mean([r['overall_overlap'] for r in results])
            avg_good_patches = np.mean([r['good_patches'] for r in results])
            
            f.write(f"📊 OVERALL STATISTICS ({len(results)} images):\n")
            f.write(f"   • Average Vessel Coverage: {avg_vessel:.1%}\n")
            f.write(f"   • Average Background Coverage: {avg_bg:.1%}\n") 
            f.write(f"   • Average Overlap: {avg_overlap:.1%}\n")
            f.write(f"   • Average Good Patches: {avg_good_patches:.1f}/64\n\n")
            
            # Per-image details
            f.write("📋 PER-IMAGE DETAILS:\n")
            f.write("-" * 30 + "\n")
            
            for result in results:
                f.write(f"\n🖼️ {result['image_name']}:\n")
                f.write(f"   • Vessel Coverage: {result['vessel_coverage']:.1%}\n")
                f.write(f"   • Background Coverage: {result['bg_coverage']:.1%}\n")
                f.write(f"   • Overall Overlap: {result['overall_overlap']:.1%}\n")
                f.write(f"   • Good Patches: {result['good_patches']}/64\n")
                f.write(f"   • Patches with Masks: {result['patches_with_masks']}/64\n")

def main():
    parser = argparse.ArgumentParser(description='Create comprehensive mask and vessel overview')
    parser.add_argument('--annotations', required=True, help='COCO annotations JSON file')
    parser.add_argument('--images', required=True, help='Images directory')
    parser.add_argument('--output-dir', default='outputs/mask_overview', help='Output directory')
    parser.add_argument('--num-images', type=int, default=5, help='Number of images to process')
    parser.add_argument('--grid-size', type=int, default=64, help='Grid cell size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Initialize generator
    generator = MaskOverviewGenerator(
        args.annotations, 
        args.images,
        args.grid_size
    )
    
    # Create batch overview
    generator.create_batch_overview(args.output_dir, args.num_images)
    
    print(f"\n🎯 **Mask Overview Generation Complete!**")
    print(f"📊 Comprehensive visualization of masks and vessels")
    print(f"🔍 Grid-based analysis with overlap detection") 
    print(f"📈 Quality assessment per 8×8 patch")

if __name__ == '__main__':
    main()