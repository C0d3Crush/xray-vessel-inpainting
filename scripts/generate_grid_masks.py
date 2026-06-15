#!/usr/bin/env python3
"""
Grid-based Background Mask Generation for Systematic Coverage

Teilt 512×512 Bilder in 8×8 Grid (64×64 Patches) und generiert für jede Zelle
eine individuelle Hintergrund-Maske für optimales Training.

Vorteile:
- 100% systematische Coverage statt zufällige Patches
- Jede 64×64 Zelle hat eigene Maske
- Bessere Datennutzung und Training-Diversität
- Kontrollierte Mask-Größen pro Patch
"""

import os
import sys
import json
import random
import argparse
import numpy as np
import cv2
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# Add src to path to import training modules
sys.path.append('src')

class GridMaskGenerator:
    def __init__(self, annotations_path, images_path, grid_size=64, safety_margin=8):
        self.annotations_path = annotations_path
        self.images_path = images_path
        self.grid_size = grid_size
        self.safety_margin = safety_margin
        
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

    def create_vessel_exclusion_mask(self, vessel_mask):
        """Create vessel exclusion zone with safety margin."""
        vessel_np = np.array(vessel_mask, dtype=np.uint8)
        
        # Create moderate safety margin - prevent overlap but allow distribution
        kernel_size = max(5, self.safety_margin + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        vessel_exclusion = cv2.dilate(vessel_np, kernel, iterations=1)  # Single iteration for minimal margin
        
        return vessel_exclusion
    
    def _create_minimal_safe_mask(self, cell_exclusion, cell_size):
        """Create minimal safe mask when cell has limited free space."""
        cell_mask = np.zeros((cell_size, cell_size), dtype=np.uint8)
        
        # Find all free pixels
        free_coords = np.where(cell_exclusion == 0)
        if len(free_coords[0]) == 0:
            print(f"    ⚠️ No free space in cell - creating empty mask")
            return cell_mask, 0
        
        # Create small circle in largest free region
        free_pixels = list(zip(free_coords[0], free_coords[1]))
        
        # Try to place a small circle
        for radius in range(3, 8):  # Small circles only
            for _ in range(20):  # Limited attempts
                if len(free_pixels) > 0:
                    center_y, center_x = random.choice(free_pixels)
                    
                    # Check if circle fits
                    temp_mask = np.zeros((cell_size, cell_size), dtype=np.uint8)
                    cv2.circle(temp_mask, (center_x, center_y), radius, 255, -1)
                    
                    # Remove vessel overlap areas
                    temp_mask[cell_exclusion > 0] = 0
                    remaining_pixels = np.sum(temp_mask > 0)
                    
                    if remaining_pixels > 0:  # Any remaining pixels after vessel removal
                        cell_mask = temp_mask
                        print(f"    ✓ Minimal safe mask: {radius}px circle ({remaining_pixels} pixels after vessel removal)")
                        return cell_mask, 1
        
        print(f"    ⚠️ Could not create safe mask - using empty")
        return cell_mask, 0

    def generate_grid_cell_mask(self, cell_x, cell_y, vessel_exclusion, max_coverage=0.35, min_coverage=0.05):
        """Generate background mask for single grid cell (64×64) - GUARANTEED mask per cell."""
        cell_size = self.grid_size
        
        # Extract vessel exclusion for this cell
        start_x = cell_x * cell_size
        start_y = cell_y * cell_size
        end_x = start_x + cell_size
        end_y = start_y + cell_size
        
        cell_exclusion = vessel_exclusion[start_y:end_y, start_x:end_x]
        
        # Check if cell has enough free space for masks
        free_pixels = np.sum(cell_exclusion == 0)  # Non-vessel pixels
        total_pixels = cell_size * cell_size
        free_ratio = free_pixels / total_pixels
        
        if free_ratio < min_coverage:
            # Not enough space for meaningful mask - create minimal safe mask
            return self._create_minimal_safe_mask(cell_exclusion, cell_size)
        
        # Generate background mask for this cell
        cell_mask = np.zeros((cell_size, cell_size), dtype=np.uint8)
        max_mask_pixels = int(cell_size * cell_size * max_coverage)
        min_mask_pixels = int(cell_size * cell_size * min_coverage)
        
        # Diverse shapes for this cell
        shapes = ['circle', 'rectangle', 'ellipse', 'triangle', 'blob', 'line', 'cross']
        successful_shapes = 0
        max_attempts_per_shape = 50
        total_attempts = 0
        
        # GUARANTEED mask approach - keep trying until we have at least min_coverage
        while True:
            current_pixels = np.sum(cell_mask > 0)
            
            # Stop if we reach max coverage limit
            if current_pixels >= max_mask_pixels:
                break
                
            # Stop if we have enough attempts and achieved minimum
            if total_attempts > 100 and current_pixels >= min_mask_pixels:
                break
                
            # Force exit if too many attempts
            if total_attempts > 200:
                print(f"    ⚠️ Cell ({cell_x},{cell_y}): Max attempts reached")
                break
                
            total_attempts += 1
            shape_type = random.choice(shapes)
            temp_mask = np.zeros((cell_size, cell_size), dtype=np.uint8)
            
            # Generate shape within cell bounds
            if shape_type == 'circle':
                radius = random.randint(4, cell_size // 6)
                center_x = random.randint(radius, cell_size - radius)
                center_y = random.randint(radius, cell_size - radius)
                cv2.circle(temp_mask, (center_x, center_y), radius, 255, -1)
                
            elif shape_type == 'rectangle':
                w = random.randint(8, cell_size // 3)
                h = random.randint(8, cell_size // 3)
                x = random.randint(0, cell_size - w)
                y = random.randint(0, cell_size - h)
                cv2.rectangle(temp_mask, (x, y), (x + w, y + h), 255, -1)
                
            elif shape_type == 'ellipse':
                center_x = random.randint(cell_size // 6, 5 * cell_size // 6)
                center_y = random.randint(cell_size // 6, 5 * cell_size // 6)
                axes_x = random.randint(4, cell_size // 8)
                axes_y = random.randint(4, cell_size // 8)
                angle = random.randint(0, 180)
                cv2.ellipse(temp_mask, (center_x, center_y), (axes_x, axes_y), 
                           angle, 0, 360, 255, -1)
                
            elif shape_type == 'triangle':
                center_x = random.randint(cell_size // 6, 5 * cell_size // 6)
                center_y = random.randint(cell_size // 6, 5 * cell_size // 6)
                size = random.randint(6, cell_size // 6)
                
                points = np.array([
                    [center_x, center_y - size],
                    [center_x - size, center_y + size//2],
                    [center_x + size, center_y + size//2]
                ], dtype=np.int32)
                cv2.fillPoly(temp_mask, [points], 255)
                
            elif shape_type == 'blob':
                # Irregular blob
                center_x = random.randint(cell_size // 6, 5 * cell_size // 6)
                center_y = random.randint(cell_size // 6, 5 * cell_size // 6)
                base_size = random.randint(4, cell_size // 8)
                
                num_points = random.randint(5, 8)
                angles = np.linspace(0, 2*np.pi, num_points+1)[:-1]
                points = []
                for angle in angles:
                    radius = base_size * random.uniform(0.5, 1.5)
                    x = int(center_x + radius * np.cos(angle))
                    y = int(center_y + radius * np.sin(angle))
                    points.append([x, y])
                
                points = np.array(points, dtype=np.int32)
                cv2.fillPoly(temp_mask, [points], 255)
                
            elif shape_type == 'line':
                # Random line within cell
                start_x = random.randint(0, cell_size)
                start_y = random.randint(0, cell_size)
                end_x = random.randint(0, cell_size)
                end_y = random.randint(0, cell_size)
                thickness = random.randint(2, 6)
                cv2.line(temp_mask, (start_x, start_y), (end_x, end_y), 255, thickness)
                
            else:  # cross
                center_x = random.randint(cell_size // 6, 5 * cell_size // 6)
                center_y = random.randint(cell_size // 6, 5 * cell_size // 6)
                arm_length = random.randint(6, cell_size // 6)
                thickness = random.randint(2, 4)
                
                # Horizontal bar
                cv2.rectangle(temp_mask, 
                            (center_x - arm_length, center_y - thickness//2),
                            (center_x + arm_length, center_y + thickness//2), 255, -1)
                # Vertical bar  
                cv2.rectangle(temp_mask, 
                            (center_x - thickness//2, center_y - arm_length),
                            (center_x + thickness//2, center_y + arm_length), 255, -1)
            
            # Check validity of this shape
            shape_pixels = np.sum(temp_mask > 0)
            if shape_pixels == 0:
                continue
                
            # Check vessel overlap
            vessel_overlap = np.sum((temp_mask > 0) & (cell_exclusion > 0))
            vessel_overlap_ratio = vessel_overlap / shape_pixels
            
            # Check existing mask overlap
            existing_overlap = np.sum((temp_mask > 0) & (cell_mask > 0))
            existing_overlap_ratio = existing_overlap / shape_pixels
            
            # Check coverage limit
            combined_mask = np.maximum(cell_mask, temp_mask)
            new_total_pixels = np.sum(combined_mask > 0)
            
            # Accept shape if good (ZERO vessel overlap required)
            if (vessel_overlap_ratio < 0.001 and  # Virtually zero vessel overlap
                existing_overlap_ratio < 0.10 and  # Minimal mask overlap
                new_total_pixels <= max_mask_pixels):
                
                cell_mask = combined_mask
                successful_shapes += 1
        
        # Final check - ensure minimum coverage is achieved
        final_pixels = np.sum(cell_mask > 0)
        if final_pixels < min_mask_pixels:
            print(f"    ⚠️ Cell ({cell_x},{cell_y}): Below minimum coverage ({final_pixels}/{min_mask_pixels})")
            print(f"      Creating emergency minimal mask...")
            emergency_mask, emergency_shapes = self._create_minimal_safe_mask(cell_exclusion, cell_size)
            if np.sum(emergency_mask > 0) > 0:
                cell_mask = np.maximum(cell_mask, emergency_mask)
                successful_shapes += emergency_shapes
                print(f"      ✓ Emergency mask added")
        
        final_coverage = (np.sum(cell_mask > 0) / (cell_size * cell_size)) * 100
        if final_coverage > 0:
            print(f"    ✓ Cell ({cell_x},{cell_y}): {final_coverage:.1f}% coverage, {successful_shapes} shapes")
        
        return cell_mask, successful_shapes
    
    def generate_guaranteed_cell_mask(self, cell_x, cell_y, vessel_exclusion, max_coverage=0.30, min_coverage=0.05):
        """Generate mask for single cell with GUARANTEED output per patch - EVERY patch gets a mask."""
        cell_size = self.grid_size
        
        # Extract vessel exclusion for this cell
        start_x = cell_x * cell_size
        start_y = cell_y * cell_size
        end_x = start_x + cell_size
        end_y = start_y + cell_size
        
        cell_exclusion = vessel_exclusion[start_y:end_y, start_x:end_x]
        
        # Calculate available space
        free_pixels = np.sum(cell_exclusion == 0)
        total_pixels = cell_size * cell_size
        free_ratio = free_pixels / total_pixels
        min_mask_pixels = int(total_pixels * min_coverage)
        
        print(f"      📊 Free space: {free_ratio:.1%} ({free_pixels}/{total_pixels} pixels)")
        print(f"      🎯 Target: minimum {min_mask_pixels} mask pixels")
        
        # Always try to create masks, even with limited space
        cell_mask = np.zeros((cell_size, cell_size), dtype=np.uint8)
        max_mask_pixels = int(total_pixels * max_coverage)
        successful_shapes = 0
        
        # PHASE 1: Try normal shapes (multiple attempts for better coverage)
        shapes = ['circle', 'rectangle', 'ellipse', 'triangle', 'line', 'blob']
        max_shape_attempts = 150  # Increased attempts
        
        for shape_attempt in range(max_shape_attempts):
            current_pixels = np.sum(cell_mask > 0)
            
            # Stop if we have enough coverage
            if current_pixels >= max_mask_pixels:
                break
                
            # Continue trying if we don't have minimum coverage yet
            shape_type = random.choice(shapes)
            temp_mask = self._generate_single_shape(shape_type, cell_size)
            
            if temp_mask is None:
                continue
                
            shape_pixels = np.sum(temp_mask > 0)
            if shape_pixels == 0:
                continue
            
            # Check vessel overlap - ZERO tolerance
            vessel_overlap = np.sum((temp_mask > 0) & (cell_exclusion > 0))
            
            # Check existing overlap
            existing_overlap = np.sum((temp_mask > 0) & (cell_mask > 0))
            existing_ratio = existing_overlap / shape_pixels
            
            # Accept only shapes with NO vessel overlap and minimal existing overlap
            if vessel_overlap == 0 and existing_ratio < 0.15:
                combined_mask = np.maximum(cell_mask, temp_mask)
                combined_pixels = np.sum(combined_mask > 0)
                
                if combined_pixels <= max_mask_pixels:
                    cell_mask = combined_mask
                    successful_shapes += 1
                    print(f"        + Added {shape_type}: {np.sum(temp_mask > 0)} pixels")
        
        current_pixels = np.sum(cell_mask > 0)
        
        # PHASE 2: If we don't have minimum coverage, force create masks
        if current_pixels < min_mask_pixels:
            print(f"      🚨 Only {current_pixels} pixels - need {min_mask_pixels}. Forcing additional masks...")
            
            # Find all completely free pixels
            free_coords = np.where((cell_exclusion == 0) & (cell_mask == 0))
            
            if len(free_coords[0]) > 0:
                # Calculate how many more pixels we need
                needed_pixels = min_mask_pixels - current_pixels
                available_pixels = len(free_coords[0])
                
                print(f"      🔧 Available free pixels: {available_pixels}, need: {needed_pixels}")
                
                if available_pixels >= needed_pixels:
                    # Randomly select pixels to fill the minimum requirement
                    indices = random.sample(range(len(free_coords[0])), min(needed_pixels, available_pixels))
                    
                    for idx in indices:
                        y, x = free_coords[0][idx], free_coords[1][idx]
                        # Create small circles around selected points
                        cv2.circle(cell_mask, (x, y), 2, 255, -1)
                    
                    # Clean up any vessel overlap (shouldn't happen but be safe)
                    cell_mask[cell_exclusion > 0] = 0
                    successful_shapes += 1
                    forced_pixels = np.sum(cell_mask > 0) - current_pixels
                    print(f"        + Forced mask: {forced_pixels} additional pixels")
        
        # PHASE 3: Final fallback - create minimal mask in center if still empty
        final_pixels = np.sum(cell_mask > 0)
        if final_pixels == 0:
            print(f"      🆘 EMERGENCY: Creating center mask")
            center_x, center_y = cell_size // 2, cell_size // 2
            
            # Try progressively larger circles from center
            for radius in range(1, 10):
                temp_mask = np.zeros((cell_size, cell_size), dtype=np.uint8)
                cv2.circle(temp_mask, (center_x, center_y), radius, 255, -1)
                
                # Remove vessel overlap
                temp_mask[cell_exclusion > 0] = 0
                remaining = np.sum(temp_mask > 0)
                
                if remaining > 0:
                    cell_mask = temp_mask
                    successful_shapes = 1
                    print(f"        🆘 Emergency: {radius}px circle, {remaining} pixels")
                    break
        
        final_pixels = np.sum(cell_mask > 0)
        final_coverage = (final_pixels / total_pixels) * 100
        
        if final_pixels > 0:
            print(f"        ✅ SUCCESS: {final_coverage:.1f}% coverage ({final_pixels} pixels)")
        else:
            print(f"        ❌ FAILED: No mask possible (vessel covers entire patch)")
        
        return cell_mask, successful_shapes
    
    def _generate_single_shape(self, shape_type, cell_size):
        """Generate a single shape for the cell."""
        temp_mask = np.zeros((cell_size, cell_size), dtype=np.uint8)
        
        if shape_type == 'circle':
            radius = random.randint(3, cell_size // 8)
            center_x = random.randint(radius, cell_size - radius - 1)
            center_y = random.randint(radius, cell_size - radius - 1)
            cv2.circle(temp_mask, (center_x, center_y), radius, 255, -1)
            
        elif shape_type == 'rectangle':
            w = random.randint(6, cell_size // 4)
            h = random.randint(6, cell_size // 4)
            x = random.randint(0, cell_size - w)
            y = random.randint(0, cell_size - h)
            cv2.rectangle(temp_mask, (x, y), (x + w, y + h), 255, -1)
            
        elif shape_type == 'ellipse':
            center_x = random.randint(cell_size // 6, 5 * cell_size // 6)
            center_y = random.randint(cell_size // 6, 5 * cell_size // 6)
            axes_x = random.randint(3, cell_size // 10)
            axes_y = random.randint(3, cell_size // 10)
            angle = random.randint(0, 180)
            cv2.ellipse(temp_mask, (center_x, center_y), (axes_x, axes_y), 
                       angle, 0, 360, 255, -1)
                       
        elif shape_type == 'triangle':
            center_x = random.randint(cell_size // 6, 5 * cell_size // 6)
            center_y = random.randint(cell_size // 6, 5 * cell_size // 6)
            size = random.randint(4, cell_size // 8)
            
            points = np.array([
                [center_x, center_y - size],
                [center_x - size, center_y + size//2],
                [center_x + size, center_y + size//2]
            ], dtype=np.int32)
            cv2.fillPoly(temp_mask, [points], 255)
            
        elif shape_type == 'line':
            # Random line within cell
            start_x = random.randint(2, cell_size - 2)
            start_y = random.randint(2, cell_size - 2)
            end_x = random.randint(2, cell_size - 2)
            end_y = random.randint(2, cell_size - 2)
            thickness = random.randint(2, 5)
            cv2.line(temp_mask, (start_x, start_y), (end_x, end_y), 255, thickness)
            
        elif shape_type == 'blob':
            # Irregular blob
            center_x = random.randint(cell_size // 6, 5 * cell_size // 6)
            center_y = random.randint(cell_size // 6, 5 * cell_size // 6)
            base_size = random.randint(3, cell_size // 10)
            
            num_points = random.randint(5, 8)
            angles = np.linspace(0, 2*np.pi, num_points+1)[:-1]
            points = []
            for angle in angles:
                radius = base_size * random.uniform(0.6, 1.4)
                x = int(center_x + radius * np.cos(angle))
                y = int(center_y + radius * np.sin(angle))
                # Clamp to cell boundaries
                x = max(0, min(cell_size-1, x))
                y = max(0, min(cell_size-1, y))
                points.append([x, y])
            
            points = np.array(points, dtype=np.int32)
            cv2.fillPoly(temp_mask, [points], 255)
        
        return temp_mask
    
    def force_minimal_mask(self, cell_x, cell_y, vessel_exclusion):
        """Force create minimal mask as last resort."""
        cell_size = self.grid_size
        
        # Extract cell exclusion
        start_x = cell_x * cell_size
        start_y = cell_y * cell_size
        end_x = start_x + cell_size
        end_y = start_y + cell_size
        
        cell_exclusion = vessel_exclusion[start_y:end_y, start_x:end_x]
        
        # Create small circle in center if possible
        cell_mask = np.zeros((cell_size, cell_size), dtype=np.uint8)
        center_x, center_y = cell_size // 2, cell_size // 2
        
        # Try small circles from center outward
        for radius in range(2, 8):
            temp_mask = np.zeros((cell_size, cell_size), dtype=np.uint8)
            cv2.circle(temp_mask, (center_x, center_y), radius, 255, -1)
            
            # Remove vessel overlap areas
            temp_mask[cell_exclusion > 0] = 0
            remaining_pixels = np.sum(temp_mask > 0)
            
            if remaining_pixels > 0:  # Any remaining pixels after vessel removal
                cell_mask = temp_mask
                print(f"        🔧 Forced minimal mask: {radius}px circle at center ({remaining_pixels} pixels)")
                break
        
        # If even center doesn't work, try random positions
        if np.sum(cell_mask) == 0:
            free_coords = np.where(cell_exclusion == 0)
            if len(free_coords[0]) > 0:
                # Pick random free pixel and place tiny circle
                idx = random.randint(0, len(free_coords[0]) - 1)
                y, x = free_coords[0][idx], free_coords[1][idx]
                temp_mask = np.zeros((cell_size, cell_size), dtype=np.uint8)
                cv2.circle(temp_mask, (x, y), 2, 255, -1)
                # Remove any vessel overlap (shouldn't be any since we picked free coords)
                temp_mask[cell_exclusion > 0] = 0
                cell_mask = temp_mask
                remaining_pixels = np.sum(cell_mask > 0)
                print(f"        🔧 Emergency 2px circle at ({x},{y}) - {remaining_pixels} pixels")
        
        return cell_mask
    
    def _final_vessel_cleanup(self, cell_mask, cell_x, cell_y, vessel_exclusion):
        """Final cleanup to ensure NO vessel overlap in saved masks."""
        cell_size = self.grid_size
        
        # Extract vessel exclusion for this cell
        start_x = cell_x * cell_size
        start_y = cell_y * cell_size
        end_x = start_x + cell_size
        end_y = start_y + cell_size
        
        cell_exclusion = vessel_exclusion[start_y:end_y, start_x:end_x]
        
        # Count original pixels
        original_pixels = np.sum(cell_mask > 0)
        
        # Remove ALL vessel pixels
        clean_mask = cell_mask.copy()
        clean_mask[cell_exclusion > 0] = 0
        
        # Count remaining pixels
        clean_pixels = np.sum(clean_mask > 0)
        removed_pixels = original_pixels - clean_pixels
        
        if removed_pixels > 0:
            print(f"      🧹 Cleanup: Removed {removed_pixels} vessel pixels ({removed_pixels/original_pixels:.1%} of mask)")
        
        return clean_mask

    def generate_grid_masks(self, image_id, output_img_dir, output_mask_dir, 
                           max_coverage_per_cell=0.35):
        """Generate all grid cell masks for one image."""
        
        # Load image
        img_info = self.id_to_info[image_id]
        img_path = os.path.join(self.images_path, img_info['file_name'])
        
        if not os.path.exists(img_path):
            print(f"❌ Image not found: {img_path}")
            return 0
            
        img = Image.open(img_path).convert('L')
        width, height = img.size
        
        if width != 512 or height != 512:
            print(f"⚠️ Image size {width}×{height} is not 512×512, skipping")
            return 0
        
        # Generate vessel masks
        vessel_mask = self.generate_vessel_mask(image_id, width, height)
        vessel_exclusion = self.create_vessel_exclusion_mask(vessel_mask)
        
        img_np = np.array(img)
        total_patches = 0
        
        # Process 6×6 inner grid (exclude outermost border cells for quality)
        grid_cells = width // self.grid_size  # Should be 8 for 512px images
        inner_grid_size = grid_cells - 2      # 6×6 inner grid (exclude border)
        
        base_name = img_info['file_name'].replace('.png', '')
        
        # Show grid layout information
        print(f"  📐 Grid Layout: {grid_cells}×{grid_cells} total grid")
        print(f"  🎯 Using inner {inner_grid_size}×{inner_grid_size} cells (excluding border)")
        print(f"  🚫 Skipping border patches for better quality")
        print(f"  ✅ Will generate {inner_grid_size * inner_grid_size} patches with GUARANTEED masks")
        
        # Skip outermost cells (start from 1, end at grid_cells-1)
        for grid_y in range(1, grid_cells-1):
            for grid_x in range(1, grid_cells-1):
                print(f"    Processing cell ({grid_x},{grid_y})...")
                
                # Extract 64×64 patch
                start_x = grid_x * self.grid_size
                start_y = grid_y * self.grid_size
                end_x = start_x + self.grid_size
                end_y = start_y + self.grid_size
                
                patch_img = img_np[start_y:end_y, start_x:end_x]
                
                # Generate mask for this specific cell with guaranteed output
                cell_mask, num_shapes = self.generate_guaranteed_cell_mask(
                    grid_x, grid_y, vessel_exclusion, max_coverage_per_cell
                )
                
                # CRITICAL: Verify mask was created - EVERY patch MUST have a mask
                mask_pixels = np.sum(cell_mask > 0)
                if mask_pixels == 0:
                    print(f"      🚨 CRITICAL: No mask created - forcing emergency mask")
                    cell_mask = self.force_minimal_mask(grid_x, grid_y, vessel_exclusion)
                    
                    # Double-check after force creation
                    final_check_pixels = np.sum(cell_mask > 0)
                    if final_check_pixels == 0:
                        print(f"      💥 ERROR: Could not create ANY mask for cell ({grid_x},{grid_y})")
                        print(f"          This patch will be SKIPPED from training!")
                        continue  # Skip saving this patch
                    else:
                        print(f"      ✅ Emergency mask created: {final_check_pixels} pixels")
                
                # Save patch and mask
                patch_id = f"{base_name}_grid_{grid_x:02d}_{grid_y:02d}"
                
                # Save patch image
                patch_pil = Image.fromarray(patch_img, 'L')
                patch_pil.save(os.path.join(output_img_dir, f"{patch_id}.png"))
                
                # Final cleanup: Remove any vessel pixels from the final mask
                cell_mask = self._final_vessel_cleanup(cell_mask, grid_x, grid_y, vessel_exclusion)
                
                # Recalculate final mask pixels after cleanup
                final_mask_pixels = np.sum(cell_mask > 0)
                
                # Save patch mask
                mask_pil = Image.fromarray(cell_mask, 'L')
                mask_pil.save(os.path.join(output_mask_dir, f"{patch_id}.png"))
                
                mask_coverage = (final_mask_pixels / (self.grid_size * self.grid_size)) * 100
                print(f"      ✓ Saved: {mask_coverage:.1f}% coverage, {num_shapes} shapes (final: {final_mask_pixels} pixels)")
                
                total_patches += 1
        
        coverage = np.sum(vessel_exclusion > 0) / (width * height)
        print(f"  📋 {img_info['file_name']}: {total_patches} patches, {coverage:.1%} vessel exclusion")
        
        return total_patches

    def process_images(self, output_img_dir, output_mask_dir, num_images=10, 
                      max_coverage_per_cell=0.35):
        """Process multiple images with grid-based mask generation."""
        
        # Create output directories
        os.makedirs(output_img_dir, exist_ok=True)
        os.makedirs(output_mask_dir, exist_ok=True)
        
        # Select random images
        selected_ids = random.sample(self.image_ids, min(num_images, len(self.image_ids)))
        
        print(f"🎯 Processing {len(selected_ids)} images with {self.grid_size}×{self.grid_size} grid cells")
        print(f"📊 Using inner 6×6 grid (excluding border patches)")
        print(f"📊 Max coverage per cell: {max_coverage_per_cell:.0%}")
        
        total_patches = 0
        
        for i, image_id in enumerate(tqdm(selected_ids, desc="Processing grid masks")):
            patches = self.generate_grid_masks(
                image_id, output_img_dir, output_mask_dir, max_coverage_per_cell
            )
            total_patches += patches
        
        print(f"\n✅ Grid processing complete!")
        print(f"  📁 Generated {total_patches} patches total")
        print(f"  🖼️ Images: {output_img_dir}/")
        print(f"  🎭 Masks:  {output_mask_dir}/")
        print(f"  📈 Patches per image: {total_patches // len(selected_ids)} (6×6 inner grid)")
        print(f"  🎲 Coverage per cell: ≤{max_coverage_per_cell:.0%}")
        print(f"  🚫 Border patches excluded for better quality")
        
        return total_patches

def main():
    parser = argparse.ArgumentParser(description='Generate grid-based background masks')
    parser.add_argument('--annotations', required=True, help='COCO annotations JSON file')
    parser.add_argument('--images', required=True, help='Images directory')
    parser.add_argument('--output-img', required=True, help='Output directory for patch images')
    parser.add_argument('--output-mask', required=True, help='Output directory for patch masks')
    parser.add_argument('--grid-size', type=int, default=64, help='Grid cell size (default: 64)')
    parser.add_argument('--num-images', type=int, default=10, help='Number of images to process')
    parser.add_argument('--max-coverage', type=float, default=0.35, help='Max mask coverage per cell')
    parser.add_argument('--safety-margin', type=int, default=12, help='Safety margin around vessels')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Initialize generator
    generator = GridMaskGenerator(
        args.annotations, 
        args.images,
        args.grid_size,
        args.safety_margin
    )
    
    # Process images
    total_patches = generator.process_images(
        args.output_img,
        args.output_mask,
        args.num_images,
        args.max_coverage
    )
    
    print(f"\n🎯 **Grid-based mask generation complete!**")
    print(f"📊 Systematic coverage: Inner 6×6 grid (excludes border)")
    print(f"🎲 Shape diversity: 7 different types per cell")
    print(f"⚖️ Controlled coverage: ≤{args.max_coverage:.0%} per {args.grid_size}×{args.grid_size} cell")
    print(f"✅ Quality focus: 36 high-quality patches per image")

if __name__ == '__main__':
    main()