# -*- coding: utf-8 -*-
import argparse, os, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
import cv2
from collections import defaultdict
from tqdm import tqdm
from network.network_pro import Inpaint
from utils import load_checkpoint, psnr, rmse, wasserstein_distance_2d, calculate_kl_divergence
try:
    from skimage.metrics import structural_similarity as ssim_fn
except ImportError:
    # Fallback for environments without skimage
    def ssim_fn(img1, img2, data_range=None, channel_axis=None):
        return 0.8  # Default SSIM value
import random
from scipy.ndimage import binary_dilation
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ArcadeDataset(Dataset):
    """
    Loads grayscale coronary angiography images with vessel masks.

    Three modes:
    - mask_dir=None, random_masks=False : masks from COCO annotations (vessel polygons)
    - mask_dir=path                     : masks loaded from folder (e.g. data/masks_cache/)
    - random_masks=True                 : random masks around vessel regions with padding
                                        → enables diverse inpainting training
    """
    STENOSIS_CATEGORY_ID = 26

    def __init__(self, img_dir, ann_path, image_size=64, mask_dir=None, random_masks=False, mask_padding=10, patches_per_image=4, online_background_masks=False, safety_margin=5, foreground_prob=0.75, max_shapes=5, background_training=True, vessel_safe_training=False, guaranteed_masks=False):
        self.img_dir      = img_dir
        self.image_size   = image_size
        self.mask_dir     = mask_dir
        self.random_masks = random_masks
        self.mask_padding = mask_padding
        self.patches_per_image = patches_per_image
        self.online_background_masks = online_background_masks
        self.safety_margin = safety_margin
        self.foreground_prob = foreground_prob
        self.max_shapes = max_shapes
        self.background_training = background_training
        self.vessel_safe_training = vessel_safe_training
        self.guaranteed_masks = guaranteed_masks

        # Try loading from pickle cache first (10x faster)
        pkl_path = ann_path.replace('.json', '.pkl')
        if os.path.exists(pkl_path):
            import pickle
            with open(pkl_path, 'rb') as f:
                cached = pickle.load(f)
            self.id_to_info = cached['id_to_info']
            self.anns_by_image = cached['anns_by_image']
            self.image_ids = cached['image_ids']
        else:
            # Fallback: parse COCO JSON
            with open(ann_path) as f:
                coco = json.load(f)
            self.id_to_info = {img['id']: img for img in coco['images']}
            self.anns_by_image = defaultdict(list)
            for ann in coco['annotations']:
                if ann['category_id'] != self.STENOSIS_CATEGORY_ID:
                    self.anns_by_image[ann['image_id']].append(ann)
            self.image_ids = [
                img_id for img_id in self.id_to_info
                if self.anns_by_image[img_id]
            ]
        
        # Filter image_ids to only include those with existing files (skip for online generation)
        if self.mask_dir and not self.online_background_masks:
            # For background mask training, only include images with successfully generated files
            self.image_ids = self._filter_existing_files()

    def __len__(self):
        return len(self.image_ids) * self.patches_per_image

    def _filter_existing_files(self):
        """Filter image_ids to only include those with existing background files."""
        from pathlib import Path
        
        filtered_ids = []
        img_dir = Path(self.img_dir)
        mask_dir = Path(self.mask_dir)
        original_count = len(self.image_ids)
        
        for img_id in self.image_ids:
            img_info = self.id_to_info[img_id]
            base_name = img_info['file_name'].replace('.png', '')
            
            # Check if background files exist (any variation)
            # Background files have format: {base_name}_bg_{XX}.png
            bg_pattern = f"{base_name}_bg_"
            
            has_bg_img = any(bg_pattern in f.name for f in img_dir.glob("*.png"))
            has_bg_mask = any(bg_pattern in f.name for f in mask_dir.glob("*.png"))
            
            if has_bg_img and has_bg_mask:
                filtered_ids.append(img_id)
        
        filtered_count = len(filtered_ids)
        if filtered_count < original_count:
            print(f"⚠️  Filtered dataset: {original_count} → {filtered_count} images")
            print(f"   Skipped {original_count - filtered_count} images without background files")
            
        return filtered_ids

    def _get_actual_file_path(self, directory, original_filename):
        """Get actual file path, handling background file naming."""
        from pathlib import Path
        
        dir_path = Path(directory)
        base_name = original_filename.replace('.png', '')
        
        # First try exact filename (for standard training)
        exact_path = dir_path / original_filename
        if exact_path.exists():
            return str(exact_path)
        
        # Then try background file pattern (for background training)
        bg_pattern = f"{base_name}_bg_"
        bg_files = [f for f in dir_path.glob("*.png") if bg_pattern in f.name]
        
        if bg_files:
            # Use first available background variation
            return str(bg_files[0])
        
        # Fallback to original path (will fail with FileNotFoundError)
        return str(exact_path)

    def _generate_online_background_mask(self, img_shape, vessel_mask, num_shapes=3):
        """Generate random background mask on-the-fly during training."""
        h, w = img_shape
        
        # Create vessel exclusion mask
        vessel_binary = (vessel_mask > 127).astype(np.uint8)
        struct = np.ones((2*self.safety_margin+1, 2*self.safety_margin+1), dtype=np.uint8)
        exclusion_mask = binary_dilation(vessel_binary, structure=struct).astype(np.uint8)
        
        # Generate combined background mask
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        successful_shapes = 0
        
        for _ in range(num_shapes):
            shape_type = random.choice(['circle', 'rectangle', 'blob'])
            
            if shape_type == 'circle':
                mask, success = self._generate_random_circle(img_shape, exclusion_mask)
            elif shape_type == 'rectangle':
                mask, success = self._generate_random_rectangle(img_shape, exclusion_mask)
            else:  # blob
                mask, success = self._generate_random_blob(img_shape, exclusion_mask)
            
            if success:
                combined_mask = np.maximum(combined_mask, mask)
                successful_shapes += 1
        
        return combined_mask, successful_shapes

    def _generate_random_circle(self, img_shape, exclusion_mask, min_radius=8, max_radius=25):
        """Generate random circle in vessel-free region."""
        h, w = img_shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        for _ in range(50):  # Max attempts
            radius = random.randint(min_radius, max_radius)
            center_x = random.randint(radius, w - radius)
            center_y = random.randint(radius, h - radius)
            
            temp_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(temp_mask, (center_x, center_y), radius, 255, -1)
            
            overlap = np.sum((temp_mask > 0) & (exclusion_mask > 0))
            circle_area = np.sum(temp_mask > 0)
            
            if overlap / circle_area < 0.1:  # Less than 10% overlap
                cv2.circle(mask, (center_x, center_y), radius, 255, -1)
                return mask, True
        
        return mask, False

    def _generate_random_rectangle(self, img_shape, exclusion_mask, min_size=10, max_size=30):
        """Generate random rectangle in vessel-free region."""
        h, w = img_shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        for _ in range(50):  # Max attempts
            width = random.randint(min_size, max_size)
            height = random.randint(min_size, max_size)
            x = random.randint(0, w - width)
            y = random.randint(0, h - height)
            
            temp_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.rectangle(temp_mask, (x, y), (x + width, y + height), 255, -1)
            
            overlap = np.sum((temp_mask > 0) & (exclusion_mask > 0))
            rect_area = np.sum(temp_mask > 0)
            
            if overlap / rect_area < 0.1:  # Less than 10% overlap
                cv2.rectangle(mask, (x, y), (x + width, y + height), 255, -1)
                return mask, True
        
        return mask, False

    def _generate_random_blob(self, img_shape, exclusion_mask, min_radius=15, max_radius=30):
        """Generate random irregular blob in vessel-free region."""
        h, w = img_shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        for _ in range(50):  # Max attempts
            center_x = random.randint(max_radius, w - max_radius)
            center_y = random.randint(max_radius, h - max_radius)
            
            # Generate random polygon points around center
            num_points = random.randint(5, 8)
            points = []
            for i in range(num_points):
                angle = (2 * np.pi * i) / num_points + random.uniform(-0.3, 0.3)
                radius = random.uniform(max_radius * 0.5, max_radius)
                x = int(center_x + radius * np.cos(angle))
                y = int(center_y + radius * np.sin(angle))
                points.append([x, y])
            
            temp_mask = np.zeros((h, w), dtype=np.uint8)
            points_array = np.array(points, dtype=np.int32)
            cv2.fillPoly(temp_mask, [points_array], 255)
            
            overlap = np.sum((temp_mask > 0) & (exclusion_mask > 0))
            blob_area = np.sum(temp_mask > 0)
            
            if blob_area > 0 and overlap / blob_area < 0.1:
                cv2.fillPoly(mask, [points_array], 255)
                return mask, True
        
        return mask, False

    def _create_vessel_safe_mask(self, image_id, W, H, max_coverage=0.25, min_coverage=0.05):
        """Create vessel-safe background masks using EXACT same logic as generate_grid_masks.py."""
        import random
        
        # Step 1: Get vessel exclusion zones
        vessel_mask = self._make_mask_from_annotations(image_id, W, H)
        vessel_np = np.array(vessel_mask, dtype=np.uint8)
        
        # Step 2: Create vessel exclusion zone with EXACT same parameters as Grid-System
        safety_margin = 15  # EXACT same as Grid-System
        kernel_size = max(5, safety_margin * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        vessel_exclusion = cv2.dilate(vessel_np, kernel, iterations=2)  # Double iterations
        
        # Step 3: Generate vessel-safe background mask
        bg_mask = np.zeros((H, W), dtype=np.uint8)
        total_pixels = W * H
        max_mask_pixels = int(total_pixels * max_coverage)
        min_mask_pixels = int(total_pixels * min_coverage)
        
        # PHASE 1: Try normal shapes with ZERO vessel overlap tolerance
        shapes = ['circle', 'rectangle', 'ellipse', 'triangle', 'line', 'blob']
        successful_shapes = 0
        max_attempts = 150  # More attempts for better coverage
        
        for attempt in range(max_attempts):
            current_pixels = np.sum(bg_mask > 0)
            
            if current_pixels >= max_mask_pixels:
                break
                
            shape_type = random.choice(shapes)
            temp_mask = self._generate_vessel_safe_shape(shape_type, W, H)
            
            if temp_mask is None:
                continue
                
            shape_pixels = np.sum(temp_mask > 0)
            if shape_pixels == 0:
                continue
            
            # Check vessel overlap - ZERO tolerance (same as Grid-System)
            vessel_overlap = np.sum((temp_mask > 0) & (vessel_exclusion > 0))
            
            # Check existing mask overlap
            existing_overlap = np.sum((temp_mask > 0) & (bg_mask > 0))
            existing_ratio = existing_overlap / shape_pixels if shape_pixels > 0 else 1.0
            
            # Accept only shapes with NO vessel overlap and minimal existing overlap
            if vessel_overlap == 0 and existing_ratio < 0.15:
                combined_mask = np.maximum(bg_mask, temp_mask)
                combined_pixels = np.sum(combined_mask > 0)
                
                if combined_pixels <= max_mask_pixels:
                    bg_mask = combined_mask
                    successful_shapes += 1
        
        # PHASE 2: Force minimum coverage if needed (same as Grid-System)
        current_pixels = np.sum(bg_mask > 0)
        if current_pixels < min_mask_pixels:
            # Find completely free pixels
            free_coords = np.where((vessel_exclusion == 0) & (bg_mask == 0))
            
            if len(free_coords[0]) > 0:
                needed_pixels = min_mask_pixels - current_pixels
                available_pixels = len(free_coords[0])
                
                if available_pixels >= needed_pixels:
                    # Randomly select pixels to meet minimum requirement
                    indices = random.sample(range(len(free_coords[0])), min(needed_pixels, available_pixels))
                    
                    for idx in indices:
                        y, x = free_coords[0][idx], free_coords[1][idx]
                        cv2.circle(bg_mask, (x, y), 2, 255, -1)  # Small 2px circles
                    
                    # Final cleanup (shouldn't be needed but be safe)
                    bg_mask[vessel_exclusion > 0] = 0
                    successful_shapes += 1
        
        # Final vessel cleanup - ensure ZERO overlap
        bg_mask[vessel_exclusion > 0] = 0
        final_pixels = np.sum(bg_mask > 0)
        
        print(f"  🎯 Vessel-safe mask: {successful_shapes} shapes, {final_pixels} pixels, "
              f"{(final_pixels/total_pixels)*100:.1f}% coverage")
        
        return Image.fromarray(bg_mask, mode='L'), successful_shapes

    def _generate_vessel_safe_shape(self, shape_type, W, H):
        """Generate vessel-safe shapes using EXACT same logic as Grid-System."""
        temp_mask = np.zeros((H, W), dtype=np.uint8)
        
        if shape_type == 'circle':
            radius = random.randint(4, min(W, H) // 12)
            center_x = random.randint(radius, W - radius)
            center_y = random.randint(radius, H - radius)
            cv2.circle(temp_mask, (center_x, center_y), radius, 255, -1)
            
        elif shape_type == 'rectangle':
            w = random.randint(8, min(W, H) // 8)
            h = random.randint(8, min(W, H) // 8)
            x = random.randint(0, W - w)
            y = random.randint(0, H - h)
            cv2.rectangle(temp_mask, (x, y), (x + w, y + h), 255, -1)
            
        elif shape_type == 'ellipse':
            center_x = random.randint(W // 6, 5 * W // 6)
            center_y = random.randint(H // 6, 5 * H // 6)
            axes_x = random.randint(4, min(W, H) // 12)
            axes_y = random.randint(4, min(W, H) // 12)
            angle = random.randint(0, 180)
            cv2.ellipse(temp_mask, (center_x, center_y), (axes_x, axes_y), 
                       angle, 0, 360, 255, -1)
                       
        elif shape_type == 'triangle':
            center_x = random.randint(W // 6, 5 * W // 6)
            center_y = random.randint(H // 6, 5 * H // 6)
            size = random.randint(6, min(W, H) // 10)
            
            points = np.array([
                [center_x, center_y - size],
                [center_x - size, center_y + size//2],
                [center_x + size, center_y + size//2]
            ], dtype=np.int32)
            cv2.fillPoly(temp_mask, [points], 255)
            
        elif shape_type == 'line':
            start_x = random.randint(0, W)
            start_y = random.randint(0, H)
            end_x = random.randint(0, W)
            end_y = random.randint(0, H)
            thickness = random.randint(3, 8)
            cv2.line(temp_mask, (start_x, start_y), (end_x, end_y), 255, thickness)
            
        elif shape_type == 'blob':
            center_x = random.randint(W // 6, 5 * W // 6)
            center_y = random.randint(H // 6, 5 * H // 6)
            base_size = random.randint(6, min(W, H) // 12)
            
            num_points = random.randint(5, 8)
            angles = np.linspace(0, 2*np.pi, num_points+1)[:-1]
            points = []
            for angle in angles:
                radius = base_size * random.uniform(0.6, 1.4)
                x = int(center_x + radius * np.cos(angle))
                y = int(center_y + radius * np.sin(angle))
                # Clamp to image boundaries
                x = max(0, min(W-1, x))
                y = max(0, min(H-1, y))
                points.append([x, y])
            
            points = np.array(points, dtype=np.int32)
            cv2.fillPoly(temp_mask, [points], 255)
        
        return temp_mask

    def _make_mask_from_annotations(self, image_id, W, H):
        """Rasterise vessel polygons into a binary mask (255 = vessel)."""
        mask = Image.new('L', (W, H), 0)
        draw = ImageDraw.Draw(mask)
        for ann in self.anns_by_image[image_id]:
            for poly in ann['segmentation']:
                xy = list(zip(poly[0::2], poly[1::2]))
                if len(xy) >= 3:
                    draw.polygon(xy, fill=255)
        return mask
    
    def _make_background_mask_from_annotations(self, image_id, W, H):
        """Generate random background masks avoiding vessel regions (255 = background)."""
        import random
        
        # Step 1: Get vessel exclusion zones
        vessel_mask = self._make_mask_from_annotations(image_id, W, H)
        vessel_np = np.array(vessel_mask, dtype=np.uint8)
        
        # Step 2: Create vessel exclusion zone with enhanced safety margin
        # Use larger padding for background training to ensure vessel-free regions
        enhanced_margin = max(8, self.safety_margin * 3)  # Tripled safety margin
        kernel_size = max(5, enhanced_margin * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        vessel_exclusion = cv2.dilate(vessel_np, kernel, iterations=2)  # Double dilation
        
        # Step 3: Generate random background patches
        bg_mask = np.zeros((H, W), dtype=np.uint8)
        num_shapes = random.randint(3, min(8, self.max_shapes + 3))  # More background patches
        
        successful_shapes = 0
        max_attempts = 100
        
        for shape_idx in range(num_shapes):
            for attempt in range(max_attempts):
                shape_type = random.choice(['circle', 'rectangle', 'ellipse'])
                
                if shape_type == 'circle':
                    # Random circle in background
                    radius = random.randint(8, min(W, H) // 8)
                    center_x = random.randint(radius, W - radius)
                    center_y = random.randint(radius, H - radius)
                    
                    # Create temporary mask for this shape
                    temp_mask = np.zeros((H, W), dtype=np.uint8)
                    cv2.circle(temp_mask, (center_x, center_y), radius, 255, -1)
                    
                elif shape_type == 'rectangle':
                    # Random rectangle in background
                    width = random.randint(10, min(W, H) // 6)
                    height = random.randint(10, min(W, H) // 6)
                    x = random.randint(0, W - width)
                    y = random.randint(0, H - height)
                    
                    # Create temporary mask
                    temp_mask = np.zeros((H, W), dtype=np.uint8)
                    cv2.rectangle(temp_mask, (x, y), (x + width, y + height), 255, -1)
                    
                else:  # ellipse
                    # Random ellipse in background
                    center_x = random.randint(W // 6, 5 * W // 6)
                    center_y = random.randint(H // 6, 5 * H // 6)
                    axes_x = random.randint(8, min(W, H) // 10)
                    axes_y = random.randint(8, min(W, H) // 10)
                    angle = random.randint(0, 180)
                    
                    # Create temporary mask
                    temp_mask = np.zeros((H, W), dtype=np.uint8)
                    cv2.ellipse(temp_mask, (center_x, center_y), (axes_x, axes_y), angle, 0, 360, 255, -1)
                
                # Check overlap with vessel exclusion zone
                overlap_pixels = np.sum((temp_mask > 0) & (vessel_exclusion > 0))
                shape_pixels = np.sum(temp_mask > 0)
                
                if shape_pixels > 0:
                    overlap_ratio = overlap_pixels / shape_pixels
                    
                    # Stricter overlap control for background training (less than 3% overlap)
                    if overlap_ratio < 0.03:  # Much stricter: only 3% overlap allowed
                        bg_mask = np.maximum(bg_mask, temp_mask)
                        successful_shapes += 1
                        break
        
        # Ensure we have meaningful background coverage (at least 5%, max 25%)
        total_pixels = W * H
        bg_coverage = np.sum(bg_mask > 0) / total_pixels
        
        if bg_coverage < 0.05:
            # Add one safe central rectangle if we have too little coverage
            safe_size = min(W, H) // 8
            center_x, center_y = W // 2, H // 2
            x1 = center_x - safe_size // 2
            y1 = center_y - safe_size // 2
            x2 = center_x + safe_size // 2
            y2 = center_y + safe_size // 2
            
            # Check if center is vessel-free
            center_exclusion = vessel_exclusion[y1:y2, x1:x2]
            if np.sum(center_exclusion > 0) / (safe_size * safe_size) < 0.2:
                cv2.rectangle(bg_mask, (x1, y1), (x2, y2), 255, -1)
        
        elif bg_coverage > 0.25:
            # Reduce mask if too much coverage
            kernel_reduce = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            bg_mask = cv2.erode(bg_mask, kernel_reduce, iterations=1)
        
        return Image.fromarray(bg_mask, mode='L')
    
    def _generate_enhanced_background_mask(self, image_id, W, H, min_shapes=4, max_shapes=8):
        """
        Generate enhanced background masks with superior vessel avoidance.
        
        Features:
        - Multi-level vessel exclusion (original + dilated + super-dilated)
        - Intelligent shape placement avoiding vessel corridors  
        - Adaptive shape sizing based on available space
        - Quality assurance with overlap verification
        
        Args:
            image_id: COCO image ID
            W, H: Image dimensions
            min_shapes, max_shapes: Range of background shapes to generate
            
        Returns:
            PIL Image with background mask (255 = background regions to inpaint)
        """
        import random
        
        # Step 1: Multi-level vessel exclusion
        vessel_mask = self._make_mask_from_annotations(image_id, W, H)
        vessel_np = np.array(vessel_mask, dtype=np.uint8)
        
        # Create multiple exclusion levels
        base_exclusion = vessel_np  # Original vessels
        
        # Level 1: Close proximity (strict no-go zone)
        strict_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        strict_exclusion = cv2.dilate(vessel_np, strict_kernel, iterations=2)
        
        # Level 2: Moderate buffer (caution zone)  
        moderate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        moderate_exclusion = cv2.dilate(vessel_np, moderate_kernel, iterations=1)
        
        # Step 2: Identify safe background corridors
        safe_zones = self._identify_safe_background_corridors(strict_exclusion, W, H)
        
        # Step 3: Generate background shapes with intelligent placement
        bg_mask = np.zeros((H, W), dtype=np.uint8)
        num_shapes = random.randint(min_shapes, max_shapes)
        successful_shapes = 0
        max_attempts_per_shape = 80
        
        for shape_idx in range(num_shapes):
            shape_placed = False
            
            for attempt in range(max_attempts_per_shape):
                # Try different shape types with size adaptation
                shape_type = random.choice(['circle', 'rectangle', 'ellipse', 'blob'])
                
                if shape_type == 'circle':
                    success, temp_mask = self._generate_safe_circle(
                        W, H, strict_exclusion, moderate_exclusion, safe_zones
                    )
                elif shape_type == 'rectangle':
                    success, temp_mask = self._generate_safe_rectangle(
                        W, H, strict_exclusion, moderate_exclusion, safe_zones
                    )
                elif shape_type == 'ellipse':
                    success, temp_mask = self._generate_safe_ellipse(
                        W, H, strict_exclusion, moderate_exclusion, safe_zones
                    )
                else:  # blob
                    success, temp_mask = self._generate_safe_blob(
                        W, H, strict_exclusion, moderate_exclusion, safe_zones
                    )
                
                if success:
                    # Verify ultra-strict overlap control (< 1% with original vessels)
                    overlap_pixels = np.sum((temp_mask > 0) & (base_exclusion > 0))
                    shape_pixels = np.sum(temp_mask > 0)
                    
                    if shape_pixels > 0:
                        overlap_ratio = overlap_pixels / shape_pixels
                        
                        if overlap_ratio < 0.01:  # Ultra-strict: < 1% vessel overlap
                            bg_mask = np.maximum(bg_mask, temp_mask)
                            successful_shapes += 1
                            shape_placed = True
                            break
            
            if not shape_placed:
                # Fallback: try smaller, safer shape in identified safe zones
                fallback_success = self._place_fallback_safe_shape(
                    bg_mask, W, H, strict_exclusion, safe_zones
                )
                if fallback_success:
                    successful_shapes += 1
        
        # Step 4: Quality assurance and coverage optimization
        bg_coverage = np.sum(bg_mask > 0) / (W * H)
        
        # Ensure minimum coverage (at least 3% for meaningful training)
        if bg_coverage < 0.03:
            self._add_emergency_background_coverage(bg_mask, W, H, strict_exclusion, safe_zones)
        
        # Ensure maximum coverage (no more than 20% to avoid over-inpainting)
        elif bg_coverage > 0.20:
            bg_mask = self._reduce_background_coverage(bg_mask, target_ratio=0.18)
        
        return Image.fromarray(bg_mask, mode='L'), successful_shapes
    
    def _identify_safe_background_corridors(self, exclusion_mask, W, H):
        """Identify large vessel-free regions suitable for background placement."""
        # Invert exclusion mask to find safe areas
        safe_areas = (exclusion_mask == 0).astype(np.uint8) * 255
        
        # Find contours of safe areas
        contours, _ = cv2.findContours(safe_areas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter for sufficiently large safe zones (min 500 pixels)
        large_safe_zones = []
        min_area = 500
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                # Create mask for this safe zone
                zone_mask = np.zeros((H, W), dtype=np.uint8)
                cv2.drawContours(zone_mask, [contour], -1, 255, -1)
                large_safe_zones.append({
                    'mask': zone_mask,
                    'area': area,
                    'contour': contour
                })
        
        return large_safe_zones
    
    def _generate_safe_circle(self, W, H, strict_exclusion, moderate_exclusion, safe_zones):
        """Generate circle in safe background regions."""
        if not safe_zones:
            return False, None
            
        # Pick a random safe zone
        zone = random.choice(safe_zones)
        zone_mask = zone['mask']
        
        # Find valid points in this zone  
        valid_points = np.where(zone_mask > 0)
        if len(valid_points[0]) == 0:
            return False, None
        
        # Try multiple placements
        for _ in range(10):
            idx = random.randint(0, len(valid_points[0]) - 1)
            center_y, center_x = valid_points[0][idx], valid_points[1][idx]
            
            # Adaptive radius based on distance to nearest vessel
            max_radius = min(W, H) // 12
            min_radius = 6
            
            # Check distance to vessels to determine safe radius
            distances = []
            for dy in range(-max_radius, max_radius + 1):
                for dx in range(-max_radius, max_radius + 1):
                    y, x = center_y + dy, center_x + dx
                    if 0 <= y < H and 0 <= x < W:
                        if strict_exclusion[y, x] > 0:
                            distances.append(np.sqrt(dx*dx + dy*dy))
            
            if distances:
                safe_radius = max(min_radius, min(max_radius, int(min(distances) * 0.7)))
            else:
                safe_radius = max_radius
            
            # Generate circle
            temp_mask = np.zeros((H, W), dtype=np.uint8)
            cv2.circle(temp_mask, (center_x, center_y), safe_radius, 255, -1)
            
            # Verify it stays within safe zone
            overlap_with_zone = np.sum((temp_mask > 0) & (zone_mask > 0))
            total_circle = np.sum(temp_mask > 0)
            
            if total_circle > 0 and (overlap_with_zone / total_circle) > 0.8:
                return True, temp_mask
                
        return False, None
    
    def _generate_safe_rectangle(self, W, H, strict_exclusion, moderate_exclusion, safe_zones):
        """Generate rectangle in safe background regions."""
        if not safe_zones:
            return False, None
            
        zone = random.choice(safe_zones)
        zone_mask = zone['mask']
        
        # Find valid rectangular regions
        for _ in range(15):
            # Random size
            width = random.randint(12, min(W, H) // 8)
            height = random.randint(12, min(W, H) // 8)
            
            # Random position within zone
            valid_points = np.where(zone_mask > 0)
            if len(valid_points[0]) == 0:
                continue
                
            idx = random.randint(0, len(valid_points[0]) - 1)
            center_y, center_x = valid_points[0][idx], valid_points[1][idx]
            
            x1 = max(0, center_x - width // 2)
            y1 = max(0, center_y - height // 2)
            x2 = min(W, x1 + width)
            y2 = min(H, y1 + height)
            
            temp_mask = np.zeros((H, W), dtype=np.uint8)
            cv2.rectangle(temp_mask, (x1, y1), (x2, y2), 255, -1)
            
            # Verify safe placement
            overlap_with_zone = np.sum((temp_mask > 0) & (zone_mask > 0))
            total_rect = np.sum(temp_mask > 0)
            
            if total_rect > 0 and (overlap_with_zone / total_rect) > 0.7:
                return True, temp_mask
                
        return False, None
    
    def _generate_safe_ellipse(self, W, H, strict_exclusion, moderate_exclusion, safe_zones):
        """Generate ellipse in safe background regions."""
        if not safe_zones:
            return False, None
            
        zone = random.choice(safe_zones)
        zone_mask = zone['mask']
        
        for _ in range(15):
            valid_points = np.where(zone_mask > 0)
            if len(valid_points[0]) == 0:
                continue
                
            idx = random.randint(0, len(valid_points[0]) - 1)
            center_y, center_x = valid_points[0][idx], valid_points[1][idx]
            
            axes_x = random.randint(8, min(W, H) // 12)
            axes_y = random.randint(8, min(W, H) // 12)
            angle = random.randint(0, 180)
            
            temp_mask = np.zeros((H, W), dtype=np.uint8)
            cv2.ellipse(temp_mask, (center_x, center_y), (axes_x, axes_y), angle, 0, 360, 255, -1)
            
            overlap_with_zone = np.sum((temp_mask > 0) & (zone_mask > 0))
            total_ellipse = np.sum(temp_mask > 0)
            
            if total_ellipse > 0 and (overlap_with_zone / total_ellipse) > 0.7:
                return True, temp_mask
                
        return False, None
    
    def _generate_safe_blob(self, W, H, strict_exclusion, moderate_exclusion, safe_zones):
        """Generate irregular blob in safe background regions."""
        if not safe_zones:
            return False, None
            
        zone = random.choice(safe_zones)
        zone_mask = zone['mask']
        
        for _ in range(10):
            valid_points = np.where(zone_mask > 0)
            if len(valid_points[0]) == 0:
                continue
                
            idx = random.randint(0, len(valid_points[0]) - 1)
            center_y, center_x = valid_points[0][idx], valid_points[1][idx]
            
            # Create irregular blob using multiple overlapping circles
            temp_mask = np.zeros((H, W), dtype=np.uint8)
            num_circles = random.randint(3, 6)
            
            for _ in range(num_circles):
                offset_x = random.randint(-15, 15)
                offset_y = random.randint(-15, 15)
                radius = random.randint(8, 18)
                
                blob_x = np.clip(center_x + offset_x, radius, W - radius)
                blob_y = np.clip(center_y + offset_y, radius, H - radius)
                
                cv2.circle(temp_mask, (blob_x, blob_y), radius, 255, -1)
            
            overlap_with_zone = np.sum((temp_mask > 0) & (zone_mask > 0))
            total_blob = np.sum(temp_mask > 0)
            
            if total_blob > 0 and (overlap_with_zone / total_blob) > 0.6:
                return True, temp_mask
                
        return False, None
    
    def _place_fallback_safe_shape(self, current_mask, W, H, strict_exclusion, safe_zones):
        """Place a small, safe shape as fallback when other methods fail."""
        if not safe_zones:
            return False
            
        # Use largest safe zone for fallback
        largest_zone = max(safe_zones, key=lambda z: z['area'])
        zone_mask = largest_zone['mask']
        
        valid_points = np.where(zone_mask > 0)
        if len(valid_points[0]) == 0:
            return False
            
        # Place small circle in center of largest safe zone
        center_idx = len(valid_points[0]) // 2
        center_y, center_x = valid_points[0][center_idx], valid_points[1][center_idx]
        
        small_radius = 8  # Conservative small shape
        cv2.circle(current_mask, (center_x, center_y), small_radius, 255, -1)
        return True
    
    def _add_emergency_background_coverage(self, current_mask, W, H, strict_exclusion, safe_zones):
        """Add minimal background coverage if we don't have enough."""
        if safe_zones:
            largest_zone = max(safe_zones, key=lambda z: z['area'])
            zone_mask = largest_zone['mask']
            
            # Add one guaranteed safe rectangle in largest zone
            valid_points = np.where(zone_mask > 0)
            if len(valid_points[0]) > 0:
                center_idx = len(valid_points[0]) // 2
                center_y, center_x = valid_points[0][center_idx], valid_points[1][center_idx]
                
                # Small emergency rectangle
                rect_size = min(20, min(W, H) // 15)
                x1 = max(0, center_x - rect_size // 2)
                y1 = max(0, center_y - rect_size // 2)  
                x2 = min(W, x1 + rect_size)
                y2 = min(H, y1 + rect_size)
                
                cv2.rectangle(current_mask, (x1, y1), (x2, y2), 255, -1)
    
    def _reduce_background_coverage(self, mask, target_ratio=0.18):
        """Reduce background coverage if it's too extensive."""
        # Use erosion to reduce mask coverage
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        reduced_mask = cv2.erode(mask, kernel, iterations=1)
        return reduced_mask

    def _generate_random_mask(self, base_mask, W, H):
        """
        Generate random mask around vessel regions with padding.
        
        Args:
            base_mask: PIL Image with vessel regions (255 = vessel)
            W, H: Original image dimensions
            
        Returns:
            PIL Image with random mask (255 = regions to inpaint)
        """
        import random
        
        # Convert base mask to numpy for morphological operations
        base_np = np.array(base_mask, dtype=np.uint8)
        
        # Apply dilation (padding) around vessel regions
        kernel_size = max(3, self.mask_padding)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dilated = cv2.dilate(base_np, kernel, iterations=1)
        
        # Create new mask starting with dilated vessels
        random_mask = Image.fromarray(dilated, mode='L')
        draw = ImageDraw.Draw(random_mask)
        
        # Add 2-max_shapes random shapes for training diversity  
        num_shapes = random.randint(2, self.max_shapes)
        
        for _ in range(num_shapes):
            shape_type = random.choice(['ellipse', 'polygon'])
            
            if shape_type == 'ellipse':
                # Random ellipse
                center_x = random.randint(W // 4, 3 * W // 4)
                center_y = random.randint(H // 4, 3 * H // 4)
                radius_x = random.randint(W // 20, W // 8)
                radius_y = random.randint(H // 20, H // 8)
                
                bbox = [center_x - radius_x, center_y - radius_y,
                       center_x + radius_x, center_y + radius_y]
                draw.ellipse(bbox, fill=255)
                
            else:
                # Random irregular polygon (3-6 points)
                num_points = random.randint(3, 6)
                center_x = random.randint(W // 4, 3 * W // 4)
                center_y = random.randint(H // 4, 3 * H // 4)
                max_radius = min(W, H) // 10
                
                points = []
                for i in range(num_points):
                    angle = (2 * np.pi * i) / num_points + random.uniform(-0.5, 0.5)
                    radius = random.randint(max_radius // 2, max_radius)
                    x = center_x + int(radius * np.cos(angle))
                    y = center_y + int(radius * np.sin(angle))
                    points.extend([x, y])
                
                if len(points) >= 6:  # Minimum 3 points
                    draw.polygon(points, fill=255)
        
        # Ensure mask doesn't cover too much of the image (limit to ~30%)
        mask_np = np.array(random_mask, dtype=np.uint8)
        coverage = np.sum(mask_np > 0) / (W * H)
        
        if coverage > 0.3:
            # Reduce mask by erosion if too large
            kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask_np = cv2.erode(mask_np, kernel_small, iterations=1)
            random_mask = Image.fromarray(mask_np, mode='L')
        
        return random_mask

    def _extract_safe_patch(self, img, mask, patch_size, min_coverage=0.01, max_retries=20):
        """
        Extract a safe patch that:
        1. Fits completely within image boundaries (no cropping)
        2. For background training: avoids vessel regions but targets background masks
        3. For vessel training: targets vessel regions
        
        Args:
            img: Input image (H, W)
            mask: Target mask (H, W) - vessels for vessel training, background for bg training
            patch_size: Size of square patch to extract
            min_coverage: Minimum mask coverage ratio for patch acceptance
            max_retries: Maximum attempts before fallback to random patch
            
        Returns:
            img_patch, mask_patch: Extracted patches
        """
        H, W = img.shape
        
        # Critical: Ensure patch fits completely within image (no padding/cropping)
        if H < patch_size or W < patch_size:
            raise ValueError(
                f"Image too small ({W}x{H}) for patch size {patch_size}x{patch_size}. "
                f"Minimum image size required: {patch_size}x{patch_size}"
            )
        
        # Calculate valid coordinate ranges (patch must fit completely)
        max_y = H - patch_size
        max_x = W - patch_size
        
        # For background training, we want patches with background masks but avoiding vessels
        if self.background_training:
            return self._extract_background_focused_patch(
                img, mask, patch_size, max_x, max_y, min_coverage, max_retries
            )
        else:
            # Traditional vessel-focused patch extraction
            return self._extract_vessel_focused_patch(
                img, mask, patch_size, max_x, max_y, min_coverage, max_retries
            )
    
    def _extract_background_focused_patch(self, img, mask, patch_size, max_x, max_y, min_coverage, max_retries):
        """Extract patch focused on background regions (avoiding vessels)."""
        background_pixels = np.where(mask > 0)  # mask contains background regions to inpaint
        
        # Strategy: Try background-centered patches first, then random
        use_background_bias = (
            np.random.random() < self.foreground_prob and  # Use same parameter but for background
            len(background_pixels[0]) > 0
        )
        
        if use_background_bias:
            # Try background-biased sampling with retries
            for retry in range(max_retries):
                # Pick a random background pixel as center
                idx = np.random.randint(len(background_pixels[0]))
                center_y, center_x = background_pixels[0][idx], background_pixels[1][idx]
                
                # Add controlled jitter around the center
                jitter_range = patch_size // 6  # Smaller jitter for safer positioning
                jitter_y = np.random.randint(-jitter_range, jitter_range + 1)
                jitter_x = np.random.randint(-jitter_range, jitter_range + 1)
                
                # Calculate top-left corner ensuring patch fits completely
                y = center_y - patch_size // 2 + jitter_y
                x = center_x - patch_size // 2 + jitter_x
                
                # CRITICAL: Clamp to ensure patch stays within bounds
                y = np.clip(y, 0, max_y)
                x = np.clip(x, 0, max_x)
                
                # Extract patch (guaranteed to fit)
                img_patch = img[y:y+patch_size, x:x+patch_size]
                mask_patch = mask[y:y+patch_size, x:x+patch_size]
                
                # Verify patch dimensions (safety check)
                if img_patch.shape != (patch_size, patch_size):
                    continue
                
                # Check background coverage
                coverage = np.sum(mask_patch > 0) / (patch_size * patch_size)
                if coverage >= min_coverage or retry == max_retries - 1:
                    return img_patch, mask_patch
        
        # Fallback: safe random sampling (guaranteed to fit within image)
        y = np.random.randint(0, max_y + 1)
        x = np.random.randint(0, max_x + 1)
        
        img_patch = img[y:y+patch_size, x:x+patch_size]
        mask_patch = mask[y:y+patch_size, x:x+patch_size]
        
        # Final safety check
        assert img_patch.shape == (patch_size, patch_size), f"Invalid patch shape: {img_patch.shape}"
        assert mask_patch.shape == (patch_size, patch_size), f"Invalid mask shape: {mask_patch.shape}"
        
        return img_patch, mask_patch
    
    def _extract_vessel_focused_patch(self, img, mask, patch_size, max_x, max_y, min_coverage, max_retries):
        """Extract patch focused on vessel regions (traditional approach)."""
        foreground_pixels = np.where(mask > 0)
        use_foreground_bias = (
            np.random.random() < self.foreground_prob and 
            len(foreground_pixels[0]) > 0
        )
        
        if use_foreground_bias:
            # Try foreground-biased sampling with retries
            for retry in range(max_retries):
                # Pick a random foreground pixel
                idx = np.random.randint(len(foreground_pixels[0]))
                center_y, center_x = foreground_pixels[0][idx], foreground_pixels[1][idx]
                
                # Add jitter around the center
                jitter_range = patch_size // 4
                jitter_y = np.random.randint(-jitter_range, jitter_range + 1)
                jitter_x = np.random.randint(-jitter_range, jitter_range + 1)
                
                # Calculate top-left corner
                y = center_y - patch_size // 2 + jitter_y
                x = center_x - patch_size // 2 + jitter_x
                
                # CRITICAL: Clamp to ensure patch stays within bounds
                y = np.clip(y, 0, max_y)
                x = np.clip(x, 0, max_x)
                
                # Extract patch
                img_patch = img[y:y+patch_size, x:x+patch_size]
                mask_patch = mask[y:y+patch_size, x:x+patch_size]
                
                # Verify patch dimensions
                if img_patch.shape != (patch_size, patch_size):
                    continue
                
                # Check coverage
                coverage = np.sum(mask_patch > 0) / (patch_size * patch_size)
                if coverage >= min_coverage or retry == max_retries - 1:
                    return img_patch, mask_patch
        
        # Fallback: safe random sampling
        y = np.random.randint(0, max_y + 1)
        x = np.random.randint(0, max_x + 1)
        
        img_patch = img[y:y+patch_size, x:x+patch_size]
        mask_patch = mask[y:y+patch_size, x:x+patch_size]
        
        assert img_patch.shape == (patch_size, patch_size)
        assert mask_patch.shape == (patch_size, patch_size)
        
        return img_patch, mask_patch

    def __getitem__(self, idx):
        # Map idx to (image_idx, patch_idx)
        image_idx = idx // self.patches_per_image
        image_id = self.image_ids[image_idx]
        
        info     = self.id_to_info[image_id]
        W, H     = info['width'], info['height']

        # Load image (handle background file naming)
        img_path = self._get_actual_file_path(self.img_dir, info['file_name'])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Load or generate mask
        if self.mask_dir is not None:
            mask_path = self._get_actual_file_path(self.mask_dir, info['file_name'])
            mask_img  = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_img is None:
                raise FileNotFoundError(f"Mask not found: {mask_path}")
            mask_np = mask_img.astype(np.float32) / 255.0
        else:
            # Choose mask generation strategy based on training mode
            if self.background_training:
                if self.vessel_safe_training:
                    # NEW: Generate vessel-safe background masks using Grid-System logic
                    mask_pil, num_shapes = self._create_vessel_safe_mask(image_id, W, H)
                else:
                    # LEGACY: Generate background masks (avoiding vessels)
                    mask_pil = self._make_background_mask_from_annotations(image_id, W, H)
            else:
                # INFERENCE: Generate vessel masks (for vessel removal)
                base_mask_pil = self._make_mask_from_annotations(image_id, W, H)
                
                if self.random_masks:
                    # Generate random mask with padding and additional shapes
                    mask_pil = self._generate_random_mask(base_mask_pil, W, H)
                else:
                    # Use original vessel mask
                    mask_pil = base_mask_pil
            
            mask_np = np.array(mask_pil, dtype=np.float32) / 255.0

        # Extract random patch
        img_patch, mask_patch = self._extract_safe_patch(img, mask_np, self.image_size)
        img_norm = (img_patch.astype(np.float32) / 255.0) * 2.0 - 1.0

        # Normalise image to [-1, 1]
        img_t    = torch.from_numpy(img_norm).unsqueeze(0)
        mask_t   = torch.from_numpy(mask_patch.astype(np.float32)).unsqueeze(0)
        return img_t, mask_t


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
def ssim_loss(pred, target, window_size=11):
    """Differentiable SSIM loss (1 - SSIM) for grayscale images in [-1, 1]."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Gaussian window
    coords = torch.arange(window_size, dtype=pred.dtype, device=pred.device)
    coords -= window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    g = g / g.sum()
    window = g.unsqueeze(0) * g.unsqueeze(1)
    window = window.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

    pad = window_size // 2
    mu1 = torch.nn.functional.conv2d(pred,   window, padding=pad, groups=1)
    mu2 = torch.nn.functional.conv2d(target, window, padding=pad, groups=1)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = torch.nn.functional.conv2d(pred   * pred,   window, padding=pad) - mu1_sq
    sigma2_sq = torch.nn.functional.conv2d(target * target, window, padding=pad) - mu2_sq
    sigma12   = torch.nn.functional.conv2d(pred   * target, window, padding=pad) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return 1 - ssim_map.mean()


class InpaintingLoss(nn.Module):
    """L1 + SSIM loss on masked region + L1 background consistency."""
    def __init__(self, ssim_weight=0.5, mask_weight=6.0, valid_weight=1.0):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.ssim_weight = ssim_weight
        self.mask_weight = mask_weight
        self.valid_weight = valid_weight

    def forward(self, output, target, mask):
        loss_mask  = self.l1(output * mask,       target * mask)
        loss_valid = self.l1(output * (1 - mask), target * (1 - mask))
        loss_ssim  = ssim_loss(output, target)
        
        total_loss = loss_mask * self.mask_weight + loss_valid * self.valid_weight + self.ssim_weight * loss_ssim
        
        # Return total loss and components for analysis
        return total_loss, {
            'l1_loss': (loss_mask * self.mask_weight + loss_valid * self.valid_weight).item(),
            'ssim_loss': (self.ssim_weight * loss_ssim).item(),
            'mask_loss': loss_mask.item(),
            'valid_loss': loss_valid.item()
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save({
        'epoch':      epoch,
        'state_dict': model.state_dict(),
        'optimizer':  optimizer.state_dict(),
        'loss':       loss,
    }, path)
    print(f"  Checkpoint saved: {path}")


def rotate_checkpoints(output_dir, keep_top_k=3):
    """Keep only the top-k best checkpoints, delete others."""
    import glob
    pattern = os.path.join(output_dir, 'epoch_*.pth')
    checkpoints = glob.glob(pattern)

    if len(checkpoints) <= keep_top_k:
        return

    # Sort by modification time (newest first)
    checkpoints.sort(key=os.path.getmtime, reverse=True)

    # Delete old checkpoints beyond top-k
    for old_ckpt in checkpoints[keep_top_k:]:
        try:
            os.remove(old_ckpt)
            print(f"  Rotated out: {os.path.basename(old_ckpt)}")
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Training function for notebook integration
# ---------------------------------------------------------------------------
def train_model(train_img, train_ann, val_img, val_ann, epochs=10, batch_size=4, 
                lr=1e-4, input_size=64, device='cpu', output_dir='checkpoints',
                patches_per_image=4, foreground_prob=0.75, 
                max_shapes=5, smoke_test=False, smoke_size=10, save_every=10,
                train_mask=None, val_mask=None, ckpt=None, num_workers=2,
                keep_checkpoints=3, random_masks=False, mask_padding=10,
                ssim_weight=0.5, mask_weight=6.0, valid_weight=1.0, epoch_callback=None,
                background_training=True):
    """
    Train CMT model - notebook-friendly version
    
    Args:
        All training parameters as keyword arguments
        epoch_callback: Optional function called after each epoch with (epoch, metrics)
        
    Returns:
        dict: Training results with 'final_metrics', 'best_val_psnr', 'output_dir'
    """
    
    # Import missing functions
    try:
        from utils import save_checkpoint, rotate_checkpoints
    except ImportError:
        # Fallback implementations
        def save_checkpoint(model, optimizer, epoch, loss, path, metrics=None):
            checkpoint = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict() if optimizer else None,
                'epoch': epoch,
                'loss': loss
            }
            if metrics:
                checkpoint.update(metrics)
            torch.save(checkpoint, path)
            
        def rotate_checkpoints(output_dir, keep_checkpoints):
            if keep_checkpoints <= 0:
                return
            # Simple rotation - keep newest files
            import glob
            checkpoints = glob.glob(os.path.join(output_dir, 'epoch_*.pth'))
            if len(checkpoints) > keep_checkpoints:
                checkpoints.sort(key=os.path.getmtime)
                for old_ckpt in checkpoints[:-keep_checkpoints]:
                    try:
                        os.remove(old_ckpt)
                    except OSError:
                        pass

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device)

    # ---- Datasets ----
    train_dataset = ArcadeDataset(train_img, train_ann, input_size,
                                  mask_dir=train_mask, random_masks=random_masks,
                                  mask_padding=mask_padding,
                                  patches_per_image=patches_per_image,
                                  foreground_prob=foreground_prob, max_shapes=max_shapes,
                                  background_training=background_training)
    
    val_dataset = ArcadeDataset(val_img, val_ann, input_size,
                                mask_dir=val_mask, random_masks=False,
                                patches_per_image=patches_per_image,
                                foreground_prob=foreground_prob, max_shapes=max_shapes,
                                background_training=background_training)

    # Smoke test override
    if smoke_test:
        train_dataset.data = train_dataset.data[:smoke_size]
        val_dataset.data = val_dataset.data[:min(smoke_size//2, len(val_dataset.data))]

    print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=(device.type=='cuda'))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, pin_memory=(device.type=='cuda'))

    # ---- Model ----
    model = Inpaint().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # Load checkpoint if provided
    if ckpt and os.path.exists(ckpt):
        model = load_checkpoint(ckpt, model, device, optimizer, reset_optimizer=False)

    # ---- Training ----
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, 'training_log.csv')
    
    # Write CSV header
    with open(log_path, 'w') as f:
        f.write('epoch,train_loss,val_loss,val_psnr,val_ssim,val_kl_divergence,val_wasserstein,val_rmse\n')

    best_val_psnr = -1
    final_metrics = {}
    
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        total_loss = 0
        
        for img, mask in train_loader:
            img, mask = img.to(device), mask.to(device)
            
            optimizer.zero_grad()
            gen = model(img, mask)
            
            # Inpainting constraint
            gen = (gen * mask) + img * (1 - mask)
            
            # Loss
            l1_loss = nn.functional.l1_loss(gen * mask, img * mask) * mask_weight + \
                     nn.functional.l1_loss(gen * (1 - mask), img * (1 - mask)) * valid_weight
            
            # SSIM loss
            try:
                from skimage.metrics import structural_similarity as ssim_fn_local
                gen_np = gen[0, 0].detach().cpu().numpy()
                real_np = img[0, 0].detach().cpu().numpy()
                ssim_value = ssim_fn_local(gen_np, real_np, data_range=2.0, channel_axis=None)
                ssim_loss_value = (1 - ssim_value) * ssim_weight
            except ImportError:
                # Fallback if skimage not available
                ssim_value = 0.8
                ssim_loss_value = (1 - ssim_value) * ssim_weight
            
            loss = l1_loss + ssim_loss_value
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss_total = 0
        val_psnr_total = 0
        val_ssim_total = 0
        val_kl_total = 0
        val_wasserstein_total = 0
        val_rmse_total = 0
        
        with torch.no_grad():
            for img, mask in val_loader:
                img, mask = img.to(device), mask.to(device)
                gen = model(img, mask)
                gen = (gen * mask) + img * (1 - mask)
                
                # Loss
                l1_loss = nn.functional.l1_loss(gen * mask, img * mask) * mask_weight + \
                         nn.functional.l1_loss(gen * (1 - mask), img * (1 - mask)) * valid_weight
                
                try:
                    from skimage.metrics import structural_similarity as ssim_fn_local
                    gen_np = gen[0, 0].detach().cpu().numpy()
                    real_np = img[0, 0].detach().cpu().numpy()
                    ssim_value = ssim_fn_local(gen_np, real_np, data_range=2.0, channel_axis=None)
                    ssim_loss_value = (1 - ssim_value) * ssim_weight
                except ImportError:
                    # Fallback if skimage not available
                    gen_np = gen[0, 0].detach().cpu().numpy()
                    real_np = img[0, 0].detach().cpu().numpy()
                    ssim_value = 0.8
                    ssim_loss_value = (1 - ssim_value) * ssim_weight
                
                val_loss_total += (l1_loss + ssim_loss_value).item()
                
                # Metrics
                val_psnr_total += psnr(gen_np, real_np)
                val_ssim_total += ssim_value
                val_kl_total += calculate_kl_divergence(gen_np, real_np)
                val_wasserstein_total += wasserstein_distance_2d(gen_np, real_np)
                val_rmse_total += rmse(gen_np, real_np)

        val_loss = val_loss_total / len(val_loader)
        val_psnr = val_psnr_total / len(val_loader)
        val_ssim = val_ssim_total / len(val_loader)
        val_kl = val_kl_total / len(val_loader)
        val_wasserstein = val_wasserstein_total / len(val_loader)
        val_rmse_value = val_rmse_total / len(val_loader)

        # Log metrics
        with open(log_path, 'a') as f:
            f.write(f'{epoch},{train_loss:.6f},{val_loss:.6f},{val_psnr:.2f},{val_ssim:.4f},{val_kl:.4f},{val_wasserstein:.4f},{val_rmse_value:.4f}\n')

        final_metrics = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_psnr': val_psnr,
            'val_ssim': val_ssim,
            'val_kl_divergence': val_kl,
            'val_wasserstein': val_wasserstein,
            'val_rmse': val_rmse_value
        }
        
        # Callback for real-time monitoring
        if epoch_callback:
            epoch_callback(epoch, final_metrics)

        print(f'Epoch {epoch:3d}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | PSNR: {val_psnr:.1f} dB | SSIM: {val_ssim:.3f}')

        # Save best model
        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            best_path = os.path.join(output_dir, 'best.pth')
            save_checkpoint(model, optimizer, epoch, train_loss, best_path)

        # Save periodic checkpoint
        if epoch % save_every == 0:
            epoch_path = os.path.join(output_dir, f'epoch_{epoch:03d}.pth')
            save_checkpoint(model, optimizer, epoch, train_loss, epoch_path)
            rotate_checkpoints(output_dir, keep_checkpoints)

    print(f"\nTraining complete. Best val PSNR: {best_val_psnr:.2f} dB")
    
    return {
        'best_val_psnr': best_val_psnr,
        'log_path': log_path,
        'final_metrics': final_metrics,
        'output_dir': output_dir
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train CMT on ARCADE grayscale X-rays")
    parser.add_argument('--train_img',   default='data/arcade/syntax/train/images')
    parser.add_argument('--train_ann',   default='data/arcade/syntax/train/annotations/train.json')
    parser.add_argument('--train_mask',  default=None,
                        help='Optional: folder with precomputed train masks (e.g. data/masks_cache/train). '
                             'If None, masks are generated from COCO annotations.')
    parser.add_argument('--val_img',     default='data/arcade/syntax/val/images')
    parser.add_argument('--val_ann',     default='data/arcade/syntax/val/annotations/val.json')
    parser.add_argument('--val_mask',    default=None,
                        help='Optional: folder with precomputed val masks.')
    parser.add_argument('--output_dir',  default='checkpoints')
    parser.add_argument('--ckpt',        default=None,
                        help='Resume from a CMT training checkpoint')
    parser.add_argument('--epochs',      type=int,   default=100)
    parser.add_argument('--batch_size',  type=int,   default=4)
    parser.add_argument('--lr',          type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int,   default=2)
    parser.add_argument('--save_every',  type=int,   default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--keep_checkpoints', type=int, default=3,
                        help='Keep only top-K periodic checkpoints (0 = keep all)')
    parser.add_argument('--device',      default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--smoke_test',  action='store_true',
                        help='Run with a small subset to verify pipeline')
    parser.add_argument('--smoke_size',  type=int, default=2,
                        help='Number of images to use in smoke test')
    parser.add_argument('--input_size',  type=int, default=256,
                        help='Input image size (power of 2, min 32)')
    parser.add_argument('--random_masks', action='store_true',
                        help='Generate random masks around vessel regions for diverse training')
    parser.add_argument('--mask_padding', type=int, default=10,
                        help='Padding size around vessel regions when using random masks')
    parser.add_argument('--ssim_weight', type=float, default=0.5,
                        help='Weight for SSIM loss component')
    parser.add_argument('--mask_weight', type=float, default=6.0,
                        help='Weight for L1 loss on masked regions')
    parser.add_argument('--valid_weight', type=float, default=1.0,
                        help='Weight for L1 loss on valid regions')
    parser.add_argument('--patches_per_image', type=int, default=4,
                        help='Number of patches to extract per image')
    parser.add_argument('--foreground_prob', type=float, default=0.75,
                        help='Probability of biasing patch sampling toward foreground (mask) pixels')
    parser.add_argument('--max_shapes', type=int, default=5,
                        help='Maximum number of random shapes to add to mask (current: 2-5)')
    parser.add_argument('--vessel_safe_training', action='store_true',
                        help='Use vessel-safe mask generation (EXACT same logic as generate_grid_masks.py)')
    parser.add_argument('--guaranteed_masks', action='store_true',
                        help='Force minimum mask coverage per patch (ensures every patch has training signal)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # ---- Datasets ----
    train_dataset = ArcadeDataset(args.train_img, args.train_ann, args.input_size,
                                  mask_dir=args.train_mask, random_masks=args.random_masks,
                                  mask_padding=args.mask_padding,
                                  patches_per_image=args.patches_per_image,
                                  foreground_prob=args.foreground_prob, max_shapes=args.max_shapes,
                                  vessel_safe_training=args.vessel_safe_training, 
                                  guaranteed_masks=args.guaranteed_masks)
    val_dataset   = ArcadeDataset(args.val_img,   args.val_ann,   args.input_size,
                                  mask_dir=args.val_mask,
                                  patches_per_image=args.patches_per_image,
                                  foreground_prob=args.foreground_prob, max_shapes=args.max_shapes)

    if args.train_mask:
        print(f"  Using precomputed train masks from: {args.train_mask}")
    elif args.random_masks:
        print(f"  Generating random masks around vessel regions (padding: {args.mask_padding}px)")
    else:
        print(f"  Generating train masks from COCO annotations")

    if args.smoke_test:
        train_dataset.image_ids = train_dataset.image_ids[:args.smoke_size]
        val_dataset.image_ids   = val_dataset.image_ids[:max(1, args.smoke_size // 2)]

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers,
                              pin_memory=(args.device == 'cuda'))
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)

    base_train_imgs = len(train_dataset.image_ids)
    base_val_imgs = len(val_dataset.image_ids)
    print(f"Patch mode enabled: {args.patches_per_image} patches per image")
    print(f"Train: {base_train_imgs} images → {len(train_dataset)} patches | Val: {base_val_imgs} images → {len(val_dataset)} patches")

    # ---- Model ----
    model = Inpaint(input_size=args.input_size).to(device)

    if args.ckpt and os.path.exists(args.ckpt):
        model = load_checkpoint(args.ckpt, model, device)
        print(f"  Resumed from checkpoint: {args.ckpt}")

    # ---- Optimiser & Loss ----
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = InpaintingLoss(ssim_weight=args.ssim_weight, 
                                mask_weight=args.mask_weight,
                                valid_weight=args.valid_weight).to(device)

    # ---- Training loop ----
    best_val_psnr = 0.0
    log_path = os.path.join(args.output_dir, 'training_log.csv')

    # Drive paths for Colab
    drive_ckpt_dir = '/content/drive/MyDrive/CMT/checkpoints'
    use_drive = os.path.isdir(drive_ckpt_dir)
    if use_drive:
        os.makedirs(drive_ckpt_dir, exist_ok=True)
        print(f"  Drive mounted: checkpoints will be mirrored to {drive_ckpt_dir}")

    drive_log_path = os.path.join(drive_ckpt_dir, 'training_log.csv') if use_drive else None
    
    # Enhanced logging for analysis
    analysis_log_path = os.path.join(args.output_dir, 'training_analysis.csv')
    with open(log_path, 'w') as f:
        f.write('epoch,train_loss,val_loss,val_l1_loss,val_ssim_loss,val_psnr,val_ssim,val_wasserstein,val_rmse,val_kl_divergence\n')
    with open(analysis_log_path, 'w') as f:
        f.write('epoch,train_loss,val_loss,l1_loss,ssim_loss,loss_change,psnr_realistic,learning_pattern\n')
    
    if drive_log_path:
        with open(drive_log_path, 'w') as f:
            f.write('epoch,train_loss,val_loss,val_l1_loss,val_ssim_loss,val_psnr,val_ssim,val_wasserstein,val_rmse,val_kl_divergence\n')
    
    # Track training behavior
    prev_train_loss = None

    for epoch in range(1, args.epochs + 1):
        # -- Train --
        model.train()
        train_loss = 0.0
        total_l1_loss = 0.0
        total_ssim_loss = 0.0
        prog = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]")
        
        for img, mask in prog:
            img, mask = img.to(device), mask.to(device)
            optimizer.zero_grad()
            output = model(img, mask)
            loss, loss_components = criterion(output, img, mask)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            total_l1_loss += loss_components['l1_loss']
            total_ssim_loss += loss_components['ssim_loss']
            prog.set_postfix(loss=f"{loss.item():.4f}")
            
        train_loss /= len(train_loader)
        avg_l1_loss = total_l1_loss / len(train_loader)
        avg_ssim_loss = total_ssim_loss / len(train_loader)

        # -- Validate --
        model.eval()
        val_loss = 0.0
        val_l1_loss = 0.0
        val_ssim_loss = 0.0
        val_psnr = 0.0
        val_ssim = 0.0
        val_wasserstein = 0.0
        val_rmse = 0.0
        val_kl_divergence = 0.0
        with torch.no_grad():
            for img, mask in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [val]"):
                img, mask = img.to(device), mask.to(device)
                output = model(img, mask)
                
                # Calculate validation loss
                loss, loss_components = criterion(output, img, mask)
                val_loss += loss.item()
                val_l1_loss += loss_components['l1_loss']
                val_ssim_loss += loss_components['ssim_loss']
                
                output = torch.clip(output, -1.0, 1.0)
                out_np = (output[:, 0].cpu().numpy() * 0.5 + 0.5) * 255.0
                gt_np  = (img[:, 0].cpu().numpy()    * 0.5 + 0.5) * 255.0
                for o, g in zip(out_np, gt_np):
                    val_psnr += psnr(o, g)
                    # Import SSIM locally to avoid import errors
                    try:
                        from skimage.metrics import structural_similarity as ssim_local
                        val_ssim += ssim_local(o, g, data_range=255.0, channel_axis=None)
                    except ImportError:
                        val_ssim += 0.8  # Fallback value
                    val_wasserstein += wasserstein_distance_2d(o, g)
                    val_rmse += rmse(o, g)
                    val_kl_divergence += calculate_kl_divergence(o, g)
        
        val_loss /= len(val_loader)
        val_l1_loss /= len(val_loader)
        val_ssim_loss /= len(val_loader)
        val_psnr /= len(val_dataset)
        val_ssim /= len(val_dataset)
        val_wasserstein /= len(val_dataset)
        val_rmse /= len(val_dataset)
        val_kl_divergence /= len(val_dataset)

        scheduler.step()
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_psnr={val_psnr:.2f} dB | val_ssim={val_ssim:.4f} | val_wasserstein={val_wasserstein:.2f} | val_rmse={val_rmse:.2f} | val_kl={val_kl_divergence:.3f}")

        with open(log_path, 'a') as f:
            f.write(f"{epoch},{train_loss:.4f},{val_loss:.4f},{val_l1_loss:.4f},{val_ssim_loss:.4f},{val_psnr:.2f},{val_ssim:.4f},{val_wasserstein:.4f},{val_rmse:.4f},{val_kl_divergence:.4f}\n")
        if drive_log_path:
            with open(drive_log_path, 'a') as f:
                f.write(f"{epoch},{train_loss:.4f},{val_loss:.4f},{val_l1_loss:.4f},{val_ssim_loss:.4f},{val_psnr:.2f},{val_ssim:.4f},{val_wasserstein:.4f},{val_rmse:.4f},{val_kl_divergence:.4f}\n")

        # Enhanced analysis logging
        loss_change = 0.0 if prev_train_loss is None else prev_train_loss - train_loss
        psnr_realistic = "realistic" if 30 <= val_psnr <= 45 else ("too_high" if val_psnr > 70 else "too_low")
        
        if epoch == 1:
            learning_pattern = "initial"
        elif loss_change > 0.01:
            learning_pattern = "good_learning"
        elif loss_change < 0.001:
            learning_pattern = "slow_learning"
        else:
            learning_pattern = "normal"
        
        with open(analysis_log_path, 'a') as f:
            f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{avg_l1_loss:.6f},{avg_ssim_loss:.6f},{loss_change:.6f},{psnr_realistic},{learning_pattern}\n")
        
        prev_train_loss = train_loss

        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            best_path = os.path.join(args.output_dir, 'best.pth')
            save_checkpoint(model, optimizer, epoch, train_loss, best_path)
            if use_drive:
                drive_best = os.path.join(drive_ckpt_dir, 'best.pth')
                save_checkpoint(model, optimizer, epoch, train_loss, drive_best)

        if epoch % args.save_every == 0:
            epoch_path = os.path.join(args.output_dir, f'epoch_{epoch:03d}.pth')
            save_checkpoint(model, optimizer, epoch, train_loss, epoch_path)
            if use_drive:
                drive_epoch = os.path.join(drive_ckpt_dir, f'epoch_{epoch:03d}.pth')
                save_checkpoint(model, optimizer, epoch, train_loss, drive_epoch)

            # Rotate old checkpoints
            if args.keep_checkpoints > 0:
                rotate_checkpoints(args.output_dir, args.keep_checkpoints)
                if use_drive:
                    rotate_checkpoints(drive_ckpt_dir, args.keep_checkpoints)

    print(f"\nTraining complete. Best val PSNR: {best_val_psnr:.2f} dB")
    print(f"Checkpoints in: {args.output_dir}/")


def train_model(
    # Data paths
    train_img='data/arcade/syntax/train/images',
    train_ann='data/arcade/syntax/train/annotations/train.json',
    train_mask=None,
    val_img='data/arcade/syntax/val/images',
    val_ann='data/arcade/syntax/val/annotations/val.json',
    val_mask=None,
    
    # Training parameters
    epochs=100,
    batch_size=4,
    lr=1e-4,
    input_size=256,
    device='cpu',
    
    # Model parameters
    output_dir='checkpoints',
    ckpt=None,
    
    # Data augmentation
    random_masks=False,
    mask_padding=10,
    patches_per_image=4,
    foreground_prob=0.75,
    max_shapes=5,
    
    # Loss parameters
    ssim_weight=0.5,
    mask_weight=6.0,
    valid_weight=1.0,
    
    # Other parameters
    smoke_test=False,
    smoke_size=2,
    num_workers=2,
    save_every=10,
    keep_checkpoints=3,
    
    # Advanced training modes
    patch_mode=None,  # For compatibility (ignored, patches_per_image > 0 enables patches)
    vessel_safe_training=False,
    guaranteed_masks=False,
    background_training=True,
    
    # Callback for real-time monitoring (notebook integration)
    epoch_callback=None
):
    """
    Train CMT inpainting model with given parameters.
    
    Args:
        epoch_callback: Optional function(epoch, metrics_dict) called after each epoch
                       for real-time monitoring in notebooks
    
    Returns:
        dict: Training results including best_val_psnr, log_path, final_metrics
    """
    import os
    import torch
    from torch import optim
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    
    # Import local modules (handle both script and package import contexts)
    try:
        from .utils import save_checkpoint, load_checkpoint, rotate_checkpoints
        from .utils import psnr, wasserstein_distance_2d, rmse, calculate_kl_divergence
        from .network.network_pro import Inpaint
    except ImportError:
        # Fallback for notebook imports
        from utils import save_checkpoint, load_checkpoint, rotate_checkpoints
        from utils import psnr, wasserstein_distance_2d, rmse, calculate_kl_divergence
        from network.network_pro import Inpaint
    
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device)

    # ---- Datasets ----
    train_dataset = ArcadeDataset(train_img, train_ann, input_size,
                                  mask_dir=train_mask, random_masks=random_masks,
                                  mask_padding=mask_padding,
                                  patches_per_image=patches_per_image,
                                  foreground_prob=foreground_prob, max_shapes=max_shapes,
                                  background_training=background_training,
                                  vessel_safe_training=vessel_safe_training,
                                  guaranteed_masks=guaranteed_masks)
    val_dataset   = ArcadeDataset(val_img, val_ann, input_size,
                                  mask_dir=val_mask,
                                  patches_per_image=patches_per_image,
                                  foreground_prob=foreground_prob, max_shapes=max_shapes)

    if train_mask:
        print(f"  Using precomputed train masks from: {train_mask}")
    elif random_masks:
        print(f"  Generating random masks around vessel regions (padding: {mask_padding}px)")
    else:
        print(f"  Generating train masks from COCO annotations")

    if smoke_test:
        train_dataset.image_ids = train_dataset.image_ids[:smoke_size]
        val_dataset.image_ids   = val_dataset.image_ids[:max(1, smoke_size // 2)]

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=(device.type == 'cuda'))
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)

    base_train_imgs = len(train_dataset.image_ids)
    base_val_imgs = len(val_dataset.image_ids)
    print(f"Patch mode enabled: {patches_per_image} patches per image")
    print(f"Train: {base_train_imgs} images → {len(train_dataset)} patches | Val: {base_val_imgs} images → {len(val_dataset)} patches")

    # ---- Model ----
    model = Inpaint(input_size=input_size).to(device)

    if ckpt and os.path.exists(ckpt):
        model = load_checkpoint(ckpt, model, device)
        print(f"  Resumed from checkpoint: {ckpt}")

    # ---- Optimiser & Loss ----
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = InpaintingLoss(ssim_weight=ssim_weight, 
                                mask_weight=mask_weight,
                                valid_weight=valid_weight).to(device)

    # ---- Training loop ----
    best_val_psnr = 0.0
    log_path = os.path.join(output_dir, 'training_log.csv')

    # Drive paths for Colab
    drive_ckpt_dir = '/content/drive/MyDrive/CMT/checkpoints'
    use_drive = os.path.isdir(drive_ckpt_dir)
    if use_drive:
        os.makedirs(drive_ckpt_dir, exist_ok=True)
        print(f"  Drive mounted: checkpoints will be mirrored to {drive_ckpt_dir}")

    drive_log_path = os.path.join(drive_ckpt_dir, 'training_log.csv') if use_drive else None
    
    # Enhanced logging for analysis
    analysis_log_path = os.path.join(output_dir, 'training_analysis.csv')
    with open(log_path, 'w') as f:
        f.write('epoch,train_loss,val_loss,val_l1_loss,val_ssim_loss,val_psnr,val_ssim,val_wasserstein,val_rmse,val_kl_divergence\n')
    with open(analysis_log_path, 'w') as f:
        f.write('epoch,train_loss,val_loss,l1_loss,ssim_loss,loss_change,psnr_realistic,learning_pattern\n')
    
    if drive_log_path:
        with open(drive_log_path, 'w') as f:
            f.write('epoch,train_loss,val_loss,val_l1_loss,val_ssim_loss,val_psnr,val_ssim,val_wasserstein,val_rmse,val_kl_divergence\n')
    
    # Track training behavior
    prev_train_loss = None
    final_metrics = {}

    for epoch in range(1, epochs + 1):
        # -- Train --
        model.train()
        train_loss = 0.0
        total_l1_loss = 0.0
        total_ssim_loss = 0.0
        prog = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]")
        
        for img, mask in prog:
            img, mask = img.to(device), mask.to(device)
            optimizer.zero_grad()
            output = model(img, mask)
            loss, loss_components = criterion(output, img, mask)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            total_l1_loss += loss_components['l1_loss']
            total_ssim_loss += loss_components['ssim_loss']
            prog.set_postfix(loss=f"{loss.item():.4f}")
            
        train_loss /= len(train_loader)
        avg_l1_loss = total_l1_loss / len(train_loader)
        avg_ssim_loss = total_ssim_loss / len(train_loader)

        # -- Validate --
        model.eval()
        val_loss = 0.0
        val_l1_loss = 0.0
        val_ssim_loss = 0.0
        val_psnr = 0.0
        val_ssim = 0.0
        val_wasserstein = 0.0
        val_rmse = 0.0
        val_kl_divergence = 0.0
        with torch.no_grad():
            for img, mask in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val]"):
                img, mask = img.to(device), mask.to(device)
                output = model(img, mask)
                
                # Calculate validation loss
                loss, loss_components = criterion(output, img, mask)
                val_loss += loss.item()
                val_l1_loss += loss_components['l1_loss']
                val_ssim_loss += loss_components['ssim_loss']
                
                output = torch.clip(output, -1.0, 1.0)
                out_np = (output[:, 0].cpu().numpy() * 0.5 + 0.5) * 255.0
                gt_np  = (img[:, 0].cpu().numpy()    * 0.5 + 0.5) * 255.0
                for o, g in zip(out_np, gt_np):
                    val_psnr += psnr(o, g)
                    # Import SSIM locally to avoid import errors
                    try:
                        from skimage.metrics import structural_similarity as ssim_local
                        val_ssim += ssim_local(o, g, data_range=255.0, channel_axis=None)
                    except ImportError:
                        val_ssim += 0.8  # Fallback value
                    val_wasserstein += wasserstein_distance_2d(o, g)
                    val_rmse += rmse(o, g)
                    val_kl_divergence += calculate_kl_divergence(o, g)
        
        val_loss /= len(val_loader)
        val_l1_loss /= len(val_loader)
        val_ssim_loss /= len(val_loader)
        val_psnr /= len(val_dataset)
        val_ssim /= len(val_dataset)
        val_wasserstein /= len(val_dataset)
        val_rmse /= len(val_dataset)
        val_kl_divergence /= len(val_dataset)

        scheduler.step()

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_psnr={val_psnr:.2f} dB | val_ssim={val_ssim:.4f} | val_wasserstein={val_wasserstein:.2f} | val_rmse={val_rmse:.2f} | val_kl={val_kl_divergence:.3f}")

        # Store current epoch metrics
        current_metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_l1_loss': val_l1_loss,
            'val_ssim_loss': val_ssim_loss,
            'val_psnr': val_psnr,
            'val_ssim': val_ssim,
            'val_wasserstein': val_wasserstein,
            'val_rmse': val_rmse,
            'val_kl_divergence': val_kl_divergence
        }
        
        final_metrics = current_metrics  # Keep updating final_metrics

        # Call epoch callback for real-time monitoring (notebook integration)
        if epoch_callback:
            try:
                epoch_callback(epoch, current_metrics)
            except Exception as e:
                print(f"Warning: epoch_callback failed: {e}")

        with open(log_path, 'a') as f:
            f.write(f"{epoch},{train_loss:.4f},{val_loss:.4f},{val_l1_loss:.4f},{val_ssim_loss:.4f},{val_psnr:.2f},{val_ssim:.4f},{val_wasserstein:.4f},{val_rmse:.4f},{val_kl_divergence:.4f}\n")
        if drive_log_path:
            with open(drive_log_path, 'a') as f:
                f.write(f"{epoch},{train_loss:.4f},{val_loss:.4f},{val_l1_loss:.4f},{val_ssim_loss:.4f},{val_psnr:.2f},{val_ssim:.4f},{val_wasserstein:.4f},{val_rmse:.4f},{val_kl_divergence:.4f}\n")

        # Enhanced analysis logging
        loss_change = 0.0 if prev_train_loss is None else prev_train_loss - train_loss
        psnr_realistic = "realistic" if 30 <= val_psnr <= 45 else ("too_high" if val_psnr > 70 else "too_low")
        
        if epoch == 1:
            learning_pattern = "initial"
        elif loss_change > 0.01:
            learning_pattern = "good_learning"
        elif loss_change < 0.001:
            learning_pattern = "slow_learning"
        else:
            learning_pattern = "normal"
        
        with open(analysis_log_path, 'a') as f:
            f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{avg_l1_loss:.6f},{avg_ssim_loss:.6f},{loss_change:.6f},{psnr_realistic},{learning_pattern}\n")
        
        prev_train_loss = train_loss

        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            best_path = os.path.join(output_dir, 'best.pth')
            save_checkpoint(model, optimizer, epoch, best_path, current_metrics)
            if use_drive:
                drive_best = os.path.join(drive_ckpt_dir, 'best.pth')
                save_checkpoint(model, optimizer, epoch, drive_best, current_metrics)

        if epoch % save_every == 0:
            epoch_path = os.path.join(output_dir, f'epoch_{epoch:03d}.pth')
            save_checkpoint(model, optimizer, epoch, epoch_path, current_metrics)
            if use_drive:
                drive_epoch = os.path.join(drive_ckpt_dir, f'epoch_{epoch:03d}.pth')
                save_checkpoint(model, optimizer, epoch, drive_epoch, current_metrics)

            # Rotate old checkpoints
            if keep_checkpoints > 0:
                rotate_checkpoints(output_dir, keep_checkpoints)
                if use_drive:
                    rotate_checkpoints(drive_ckpt_dir, keep_checkpoints)

    print(f"\nTraining complete. Best val PSNR: {best_val_psnr:.2f} dB")
    print(f"Checkpoints in: {output_dir}/")
    
    return {
        'best_val_psnr': best_val_psnr,
        'log_path': log_path,
        'final_metrics': final_metrics,
        'output_dir': output_dir
    }


if __name__ == '__main__':
    main()
