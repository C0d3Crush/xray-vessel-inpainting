#!/usr/bin/env python3
"""
Visualisierung des Mask-Problems: Zeigt Gefäße vs. Training-Masks vs. Patches

Zeigt das Problem auf:
- ROT: Original Gefäße (die wir NICHT inpainten wollen)
- BLAU: Background-Training-Masks (die wir inpainten wollen) 
- GRÜN: Patch-Grenzen (müssen komplett im Bild sein)
- GELB: Gefährliche Überlappungen (Patches die Gefäße schneiden)

Erstellt Side-by-Side Vergleich: Aktuell vs. Korrekt
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
from pathlib import Path

# Add src to path to import training modules
sys.path.append('src')

class MaskProblemVisualizer:
    def __init__(self, annotations_path, images_path):
        self.annotations_path = annotations_path
        self.images_path = images_path
        
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
        """Generate vessel mask from COCO annotations (ROT)."""
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
    
    def generate_background_mask(self, vessel_mask, width, height, safety_margin=8, num_shapes=5):
        """Generate safe background mask avoiding vessels (BLAU)."""
        vessel_np = np.array(vessel_mask, dtype=np.uint8)
        
        # Create vessel exclusion zone with safety margin
        kernel_size = max(3, safety_margin * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        vessel_exclusion = cv2.dilate(vessel_np, kernel, iterations=1)
        
        # Generate random background patches
        bg_mask = np.zeros((height, width), dtype=np.uint8)
        successful_shapes = 0
        max_attempts = 100
        
        for shape_idx in range(num_shapes):
            for attempt in range(max_attempts):
                shape_type = random.choice(['circle', 'rectangle', 'ellipse'])
                
                if shape_type == 'circle':
                    radius = random.randint(8, min(width, height) // 8)
                    center_x = random.randint(radius, width - radius)
                    center_y = random.randint(radius, height - radius)
                    
                    temp_mask = np.zeros((height, width), dtype=np.uint8)
                    cv2.circle(temp_mask, (center_x, center_y), radius, 255, -1)
                    
                elif shape_type == 'rectangle':
                    w = random.randint(10, min(width, height) // 6)
                    h = random.randint(10, min(width, height) // 6)
                    x = random.randint(0, width - w)
                    y = random.randint(0, height - h)
                    
                    temp_mask = np.zeros((height, width), dtype=np.uint8)
                    cv2.rectangle(temp_mask, (x, y), (x + w, y + h), 255, -1)
                    
                else:  # ellipse
                    center_x = random.randint(width // 6, 5 * width // 6)
                    center_y = random.randint(height // 6, 5 * height // 6)
                    axes_x = random.randint(8, min(width, height) // 10)
                    axes_y = random.randint(8, min(width, height) // 10)
                    angle = random.randint(0, 180)
                    
                    temp_mask = np.zeros((height, width), dtype=np.uint8)
                    cv2.ellipse(temp_mask, (center_x, center_y), (axes_x, axes_y), 
                               angle, 0, 360, 255, -1)
                
                # Check overlap with vessel exclusion zone
                overlap_pixels = np.sum((temp_mask > 0) & (vessel_exclusion > 0))
                shape_pixels = np.sum(temp_mask > 0)
                
                if shape_pixels > 0:
                    overlap_ratio = overlap_pixels / shape_pixels
                    
                    # Accept shape if less than 10% overlap with vessels
                    if overlap_ratio < 0.1:
                        bg_mask = np.maximum(bg_mask, temp_mask)
                        successful_shapes += 1
                        break
        
        return Image.fromarray(bg_mask, mode='L'), successful_shapes
    
    def extract_safe_patches(self, img_shape, patch_size, num_patches=8):
        """Extrahiere Patches die komplett im Bild liegen (GRÜN)."""
        height, width = img_shape
        patches = []
        
        # Ensure patches fit completely within image
        max_y = height - patch_size
        max_x = width - patch_size
        
        if max_x < 0 or max_y < 0:
            print(f"⚠️  Image too small ({width}x{height}) for patch size {patch_size}")
            return []
        
        for _ in range(num_patches):
            # Random position ensuring patch fits completely
            y = random.randint(0, max_y)
            x = random.randint(0, max_x)
            patches.append((x, y, x + patch_size, y + patch_size))
            
        return patches
    
    def extract_dangerous_patches(self, img_shape, patch_size, num_patches=4):
        """Extrahiere gefährliche Patches die am Rand abgeschnitten sind (GELB)."""
        height, width = img_shape
        patches = []
        
        for _ in range(num_patches):
            # Allow patches to go partially outside image bounds
            margin = patch_size // 3
            y = random.randint(-margin, height - patch_size + margin)
            x = random.randint(-margin, width - patch_size + margin)
            patches.append((x, y, x + patch_size, y + patch_size))
            
        return patches
    
    def check_vessel_patch_overlap(self, vessel_mask, patch_coords):
        """Prüfe ob Patch gefährlich Gefäße schneidet."""
        vessel_np = np.array(vessel_mask, dtype=np.uint8)
        height, width = vessel_np.shape
        
        overlaps = []
        for x1, y1, x2, y2 in patch_coords:
            # Clamp to image boundaries
            x1_c = max(0, min(x1, width))
            y1_c = max(0, min(y1, height))
            x2_c = max(0, min(x2, width))
            y2_c = max(0, min(y2, height))
            
            if x2_c <= x1_c or y2_c <= y1_c:
                overlaps.append(0.0)  # No valid area
                continue
                
            # Extract patch region from vessel mask
            patch_vessel = vessel_np[y1_c:y2_c, x1_c:x2_c]
            
            if patch_vessel.size == 0:
                overlaps.append(0.0)
                continue
                
            # Calculate overlap ratio
            vessel_pixels = np.sum(patch_vessel > 0)
            total_pixels = patch_vessel.size
            overlap_ratio = vessel_pixels / total_pixels if total_pixels > 0 else 0.0
            overlaps.append(overlap_ratio)
            
        return overlaps
    
    def create_visualization(self, image_id, patch_size=64, output_path=None):
        """Erstelle komplette Visualisierung des Mask-Problems."""
        # Load image
        img_info = self.id_to_info[image_id]
        img_path = os.path.join(self.images_path, img_info['file_name'])
        
        if not os.path.exists(img_path):
            print(f"❌ Image not found: {img_path}")
            return None
            
        img = Image.open(img_path).convert('L')
        width, height = img.size
        
        print(f"🖼️  Processing {img_info['file_name']} ({width}x{height})")
        
        # Generate masks
        vessel_mask = self.generate_vessel_mask(image_id, width, height)
        bg_mask, num_shapes = self.generate_background_mask(vessel_mask, width, height)
        
        print(f"🎭 Generated background mask with {num_shapes} shapes")
        
        # Generate patches
        safe_patches = self.extract_safe_patches((height, width), patch_size, num_patches=6)
        dangerous_patches = self.extract_dangerous_patches((height, width), patch_size, num_patches=3)
        
        # Check overlaps
        safe_overlaps = self.check_vessel_patch_overlap(vessel_mask, safe_patches)
        dangerous_overlaps = self.check_vessel_patch_overlap(vessel_mask, dangerous_patches)
        
        print(f"✅ Safe patches: {len(safe_patches)} (avg overlap: {np.mean(safe_overlaps):.2%})")
        print(f"⚠️  Dangerous patches: {len(dangerous_patches)} (avg overlap: {np.mean(dangerous_overlaps):.2%})")
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Mask Problem Visualization: {img_info["file_name"]}\\n'
                    f'Patch Size: {patch_size}×{patch_size}, Image: {width}×{height}', 
                    fontsize=16, fontweight='bold')
        
        # Convert masks to RGB for colored overlay
        img_rgb = np.stack([np.array(img)] * 3, axis=-1)
        vessel_rgb = np.array(vessel_mask)
        bg_rgb = np.array(bg_mask)
        
        # 1. Problem Visualization (Current)
        ax1 = axes[0, 0]
        ax1.imshow(img_rgb)
        ax1.imshow(vessel_rgb, alpha=0.4, cmap='Reds')  # ROT: Gefäße
        ax1.imshow(bg_rgb, alpha=0.3, cmap='Blues')     # BLAU: Background masks
        
        # Add dangerous patches (GELB)
        for i, (x1, y1, x2, y2) in enumerate(dangerous_patches):
            overlap = dangerous_overlaps[i]
            color = 'yellow' if overlap > 0.05 else 'orange'
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   linewidth=2, edgecolor=color, facecolor='none', alpha=0.8)
            ax1.add_patch(rect)
            ax1.text(x1+2, y1+12, f'{overlap:.1%}', color=color, fontweight='bold', fontsize=8)
            
        ax1.set_title('❌ Problem: Gefährliche Patches (Aktuell)', fontweight='bold', color='red')
        ax1.set_xlabel('Gelb: Patches schneiden Gefäße | Rot: Gefäße | Blau: Training Masks')
        ax1.axis('off')
        
        # 2. Solution Visualization (Corrected)
        ax2 = axes[0, 1]
        ax2.imshow(img_rgb)
        ax2.imshow(vessel_rgb, alpha=0.4, cmap='Reds')  # ROT: Gefäße
        ax2.imshow(bg_rgb, alpha=0.3, cmap='Blues')     # BLAU: Background masks
        
        # Add safe patches (GRÜN)
        for i, (x1, y1, x2, y2) in enumerate(safe_patches):
            overlap = safe_overlaps[i]
            color = 'green' if overlap < 0.05 else 'red'
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   linewidth=2, edgecolor=color, facecolor='none', alpha=0.8)
            ax2.add_patch(rect)
            ax2.text(x1+2, y1+12, f'{overlap:.1%}', color=color, fontweight='bold', fontsize=8)
            
        ax2.set_title('✅ Lösung: Sichere Patches (Korrekt)', fontweight='bold', color='green')
        ax2.set_xlabel('Grün: Sichere Patches | Rot: Gefäße | Blau: Training Masks')
        ax2.axis('off')
        
        # 3. Vessel Exclusion Analysis
        ax3 = axes[1, 0]
        vessel_np = np.array(vessel_mask, dtype=np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))  # 8*2+1
        vessel_exclusion = cv2.dilate(vessel_np, kernel, iterations=1)
        
        ax3.imshow(img_rgb)
        ax3.imshow(vessel_np, alpha=0.6, cmap='Reds')
        ax3.imshow(vessel_exclusion, alpha=0.3, cmap='Oranges')
        ax3.set_title('🛡️ Gefäß-Exclusion-Zone (Safety Margin)', fontweight='bold')
        ax3.set_xlabel('Rot: Original Gefäße | Orange: Safety Padding')
        ax3.axis('off')
        
        # 4. Coverage Analysis
        ax4 = axes[1, 1]
        vessel_coverage = np.sum(vessel_np > 0) / (width * height)
        bg_coverage = np.sum(bg_rgb > 0) / (width * height)
        exclusion_coverage = np.sum(vessel_exclusion > 0) / (width * height)
        
        coverage_data = {
            'Gefäße': vessel_coverage,
            'Safety Zone': exclusion_coverage, 
            'Background Masks': bg_coverage,
            'Verfügbar': 1.0 - exclusion_coverage
        }
        
        colors = ['red', 'orange', 'blue', 'lightgreen']
        wedges, texts, autotexts = ax4.pie(coverage_data.values(), 
                                          labels=coverage_data.keys(),
                                          autopct='%1.1f%%',
                                          colors=colors,
                                          startangle=90)
        
        ax4.set_title('📊 Flächen-Verteilung', fontweight='bold')
        
        # Patch overlap statistics
        safe_avg = np.mean(safe_overlaps)
        danger_avg = np.mean(dangerous_overlaps)
        
        stats_text = (f"Patch-Statistiken ({patch_size}×{patch_size}):\\n"
                     f"✅ Sichere Patches: {safe_avg:.1%} Gefäß-Overlap\\n"
                     f"⚠️ Gefährliche Patches: {danger_avg:.1%} Gefäß-Overlap\\n"
                     f"🎯 Ziel: <5% Overlap für Background-Training")
        
        fig.text(0.02, 0.02, stats_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"💾 Saved visualization: {output_path}")
        else:
            plt.show()
            
        plt.close()
        
        # Return analysis results
        return {
            'vessel_coverage': vessel_coverage,
            'bg_coverage': bg_coverage,
            'safe_patch_overlap': safe_avg,
            'dangerous_patch_overlap': danger_avg,
            'num_bg_shapes': num_shapes
        }

def main():
    parser = argparse.ArgumentParser(description='Visualize mask problem for background inpainting')
    parser.add_argument('--annotations', required=True, help='Path to COCO annotations JSON')
    parser.add_argument('--images', required=True, help='Path to images directory')
    parser.add_argument('--output-dir', default='outputs/mask_analysis', help='Output directory')
    parser.add_argument('--num-samples', type=int, default=3, help='Number of sample images to process')
    parser.add_argument('--patch-sizes', nargs='+', type=int, default=[32, 64, 128], 
                       help='Patch sizes to test')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize visualizer
    visualizer = MaskProblemVisualizer(args.annotations, args.images)
    
    if len(visualizer.image_ids) == 0:
        print("❌ No annotated images found!")
        return
    
    # Process sample images
    sample_ids = random.sample(visualizer.image_ids, min(args.num_samples, len(visualizer.image_ids)))
    
    print(f"🎯 Processing {len(sample_ids)} sample images with patch sizes {args.patch_sizes}")
    
    all_results = []
    
    for i, image_id in enumerate(sample_ids):
        img_info = visualizer.id_to_info[image_id]
        base_name = img_info['file_name'].replace('.png', '')
        
        print(f"\\n📷 [{i+1}/{len(sample_ids)}] {img_info['file_name']}")
        
        for patch_size in args.patch_sizes:
            output_path = os.path.join(args.output_dir, 
                                     f"{base_name}_patch{patch_size}_analysis.png")
            
            result = visualizer.create_visualization(image_id, patch_size, output_path)
            
            if result:
                result['image_name'] = img_info['file_name']
                result['patch_size'] = patch_size
                all_results.append(result)
    
    # Summary statistics
    print(f"\\n📊 ZUSAMMENFASSUNG ({len(all_results)} Analysen):")
    print("=" * 60)
    
    for patch_size in args.patch_sizes:
        results_for_size = [r for r in all_results if r['patch_size'] == patch_size]
        if results_for_size:
            avg_safe = np.mean([r['safe_patch_overlap'] for r in results_for_size])
            avg_danger = np.mean([r['dangerous_patch_overlap'] for r in results_for_size])
            avg_bg = np.mean([r['bg_coverage'] for r in results_for_size])
            
            print(f"Patch {patch_size}×{patch_size}:")
            print(f"  ✅ Sichere Patches:     {avg_safe:.1%} Gefäß-Overlap")
            print(f"  ⚠️  Gefährliche Patches: {avg_danger:.1%} Gefäß-Overlap")
            print(f"  🎭 Background Coverage:  {avg_bg:.1%}")
            print(f"  📈 Verbesserung:         {(avg_danger-avg_safe)/avg_danger*100:.0f}%")
            print()
    
    print(f"🎯 **Empfehlung:** Verwende sichere Patch-Extraktion für Background-Training!")
    print(f"📁 Alle Visualisierungen gespeichert in: {args.output_dir}")

if __name__ == '__main__':
    main()