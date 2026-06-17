# -*- coding: utf-8 -*-
import os, json, random
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import cv2
from utils import load_coco_annotations


class ArcadeDataset(Dataset):
    """
    Loads grayscale coronary angiography images with vessel masks.

    Training modes (controlled by parameter combination):
    - mask_dir=path              : load precomputed masks from folder (fastest)
    - vessel_safe_training=True  : generate vessel-safe background masks on-the-fly
    - background_training=True   : generate background masks avoiding vessel regions
    - background_training=False  : generate vessel masks from COCO (for vessel removal)
    - random_masks=True          : dilated vessel masks + random shapes for diversity
    """
    STENOSIS_CATEGORY_ID = 26

    def __init__(
        self,
        img_dir: str,
        ann_path: str,
        *,
        # patch extraction
        image_size: int = 64,
        patches_per_image: int = 4,
        foreground_prob: float = 0.75,
        # mask source
        mask_dir: str = None,
        random_masks: bool = False,
        mask_padding: int = 10,
        # training mode
        background_training: bool = True,
        vessel_safe_training: bool = False,
        # shape generation
        safety_margin: int = 5,
        max_shapes: int = 5,
    ):
        if not os.path.isdir(img_dir):
            raise NotADirectoryError(f"img_dir not found: {img_dir}")
        if not os.path.isfile(ann_path):
            raise FileNotFoundError(f"ann_path not found: {ann_path}")
        if mask_dir is not None and not os.path.isdir(mask_dir):
            raise NotADirectoryError(f"mask_dir not found: {mask_dir}")
        if not 0.0 <= foreground_prob <= 1.0:
            raise ValueError(f"foreground_prob must be in [0, 1], got {foreground_prob}")
        if image_size < 32:
            raise ValueError(f"image_size must be >= 32, got {image_size}")
        if patches_per_image < 1:
            raise ValueError(f"patches_per_image must be >= 1, got {patches_per_image}")

        self.img_dir      = img_dir
        self.image_size   = image_size
        self.mask_dir     = mask_dir
        self.random_masks = random_masks
        self.mask_padding = mask_padding
        self.patches_per_image = patches_per_image
        self.safety_margin = safety_margin
        self.foreground_prob = foreground_prob
        self.max_shapes = max_shapes
        self.background_training = background_training
        self.vessel_safe_training = vessel_safe_training
        self._img_cache  = self._build_path_cache(self.img_dir)
        self._mask_cache = self._build_path_cache(self.mask_dir) if self.mask_dir else {}

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
            self.id_to_info, self.anns_by_image, self.image_ids = load_coco_annotations(ann_path)

        if self.mask_dir:
            self.image_ids = self._filter_existing_files()

    def __len__(self):
        return len(self.image_ids) * self.patches_per_image

    def _filter_existing_files(self):
        """Filter image_ids to only those with matching files in both img and mask dirs."""
        original_count = len(self.image_ids)
        filtered_ids = [
            img_id for img_id in self.image_ids
            if os.path.splitext(self.id_to_info[img_id]['file_name'])[0] in self._img_cache
            and os.path.splitext(self.id_to_info[img_id]['file_name'])[0] in self._mask_cache
        ]
        filtered_count = len(filtered_ids)
        if filtered_count < original_count:
            print(f"⚠️  Filtered dataset: {original_count} → {filtered_count} images")
            print(f"   Skipped {original_count - filtered_count} images without precomputed files")
        return filtered_ids

    def _build_path_cache(self, directory: str) -> dict:
        cache = {}
        for f in Path(directory).glob("*.png"):
            stem = f.stem
            if stem not in cache:
                cache[stem] = str(f)
            if "_bg_" in stem:
                base = stem.split("_bg_")[0]
                if base not in cache:
                    cache[base] = str(f)
        return cache

    def _get_actual_file_path(self, directory: str, original_filename: str) -> str:
        base_name = original_filename.replace('.png', '')
        cache = self._img_cache if directory == self.img_dir else self._mask_cache
        if base_name in cache:
            return cache[base_name]
        return os.path.join(directory, original_filename)

    def _generate_vessel_free_mask(self, image_id, W, H,
                                    dilation_margin, overlap_tolerance,
                                    shape_types, max_attempts,
                                    min_coverage=0.05, max_coverage=0.25):
        """
        Shared core for vessel-free mask generation.

        Dilates vessel annotations by dilation_margin px, then fills the safe
        area with random shapes.  Returns (PIL mask, num_successful_shapes).
        """
        vessel_np = np.array(self._make_mask_from_annotations(image_id, W, H), dtype=np.uint8)
        kernel_size = max(5, dilation_margin * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        vessel_exclusion = cv2.dilate(vessel_np, kernel, iterations=2)

        bg_mask = np.zeros((H, W), dtype=np.uint8)
        total_pixels = W * H
        max_mask_pixels = int(total_pixels * max_coverage)
        min_mask_pixels = int(total_pixels * min_coverage)
        successful_shapes = 0

        for _ in range(max_attempts):
            if np.sum(bg_mask > 0) >= max_mask_pixels:
                break
            shape_type = random.choice(shape_types)
            temp_mask = self._generate_vessel_safe_shape(shape_type, W, H)
            shape_pixels = np.sum(temp_mask > 0)
            if shape_pixels == 0:
                continue
            vessel_overlap = np.sum((temp_mask > 0) & (vessel_exclusion > 0))
            existing_ratio = np.sum((temp_mask > 0) & (bg_mask > 0)) / shape_pixels
            if vessel_overlap / shape_pixels <= overlap_tolerance and existing_ratio < 0.15:
                combined = np.maximum(bg_mask, temp_mask)
                if np.sum(combined > 0) <= max_mask_pixels:
                    bg_mask = combined
                    successful_shapes += 1

        # Force minimum coverage using small scatter circles in vessel-free area
        if np.sum(bg_mask > 0) < min_mask_pixels:
            free_coords = np.where((vessel_exclusion == 0) & (bg_mask == 0))
            needed = min_mask_pixels - np.sum(bg_mask > 0)
            if len(free_coords[0]) >= needed:
                for idx in random.sample(range(len(free_coords[0])), needed):
                    cv2.circle(bg_mask, (int(free_coords[1][idx]), int(free_coords[0][idx])), 2, 255, -1)
                successful_shapes += 1

        bg_mask[vessel_exclusion > 0] = 0
        return Image.fromarray(bg_mask, mode='L'), successful_shapes

    def _create_vessel_safe_mask(self, image_id, W, H, max_coverage=0.25, min_coverage=0.05):
        """Vessel-safe mask with zero overlap tolerance (matches grid-system parameters)."""
        return self._generate_vessel_free_mask(
            image_id, W, H,
            dilation_margin=15,
            overlap_tolerance=0.0,
            shape_types=['circle', 'rectangle', 'ellipse', 'triangle', 'line', 'blob'],
            max_attempts=150,
            min_coverage=min_coverage,
            max_coverage=max_coverage,
        )

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
            start_x = random.randint(0, W - 1)
            start_y = random.randint(0, H - 1)
            end_x = random.randint(0, W - 1)
            end_y = random.randint(0, H - 1)
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


    def _generate_random_mask(self, base_mask, W, H):
        """
        Generate random mask around vessel regions with padding.

        Args:
            base_mask: PIL Image with vessel regions (255 = vessel)
            W, H: Original image dimensions

        Returns:
            PIL Image with random mask (255 = regions to inpaint)
        """
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

                coverage = np.sum(mask_patch > 0) / (patch_size * patch_size)
                if coverage >= min_coverage:
                    return img_patch, mask_patch

        # Fallback: safe random sampling
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

                coverage = np.sum(mask_patch > 0) / (patch_size * patch_size)
                if coverage >= min_coverage:
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
                    mask_pil, _ = self._create_vessel_safe_mask(image_id, W, H)
                else:
                    mask_pil, _ = self._create_vessel_safe_mask(image_id, W, H)
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
