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
from utils import load_checkpoint, psnr, rmse, wasserstein_distance_2d
from skimage.metrics import structural_similarity as ssim_fn
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

    def __init__(self, img_dir, ann_path, image_size=256, mask_dir=None, random_masks=False, mask_padding=10, patch_mode=False, patches_per_image=4, online_background_masks=False, safety_margin=5, foreground_prob=0.75):
        self.img_dir      = img_dir
        self.image_size   = image_size
        self.mask_dir     = mask_dir
        self.random_masks = random_masks
        self.mask_padding = mask_padding
        self.patch_mode   = patch_mode
        self.patches_per_image = patches_per_image
        self.online_background_masks = online_background_masks
        self.safety_margin = safety_margin
        self.foreground_prob = foreground_prob

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
        if self.patch_mode:
            return len(self.image_ids) * self.patches_per_image
        return len(self.image_ids)

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
        
        # Add 2-5 random shapes for training diversity
        num_shapes = random.randint(2, 5)
        
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

    def _extract_random_patch(self, img, mask, patch_size, min_coverage=0.01, max_retries=5):
        """Extract a patch with foreground bias to ensure mask coverage."""
        H, W = img.shape
        
        # Ensure patch fits within image
        if H < patch_size or W < patch_size:
            # Pad image if smaller than patch size
            pad_h = max(0, patch_size - H)
            pad_w = max(0, patch_size - W)
            img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
            H, W = img.shape
        
        max_y = H - patch_size
        max_x = W - patch_size
        
        # Check if we should use foreground-biased sampling
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
                
                # Add jitter around the center (±patch_size//4)
                jitter_range = patch_size // 4
                jitter_y = np.random.randint(-jitter_range, jitter_range + 1)
                jitter_x = np.random.randint(-jitter_range, jitter_range + 1)
                
                # Calculate top-left corner (center - patch_size//2 + jitter)
                y = center_y - patch_size // 2 + jitter_y
                x = center_x - patch_size // 2 + jitter_x
                
                # Clamp to valid range
                y = np.clip(y, 0, max_y)
                x = np.clip(x, 0, max_x)
                
                # Extract patch
                img_patch = img[y:y+patch_size, x:x+patch_size]
                mask_patch = mask[y:y+patch_size, x:x+patch_size]
                
                # Check coverage
                coverage = np.sum(mask_patch > 0) / (patch_size * patch_size)
                if coverage >= min_coverage or retry == max_retries - 1:
                    return img_patch, mask_patch
        
        # Fallback: random sampling (either by choice or if foreground bias failed)
        y = np.random.randint(0, max_y + 1)
        x = np.random.randint(0, max_x + 1)
        
        img_patch = img[y:y+patch_size, x:x+patch_size]
        mask_patch = mask[y:y+patch_size, x:x+patch_size]
        
        return img_patch, mask_patch

    def __getitem__(self, idx):
        if self.patch_mode:
            # In patch mode, map idx to (image_idx, patch_idx)
            image_idx = idx // self.patches_per_image
            image_id = self.image_ids[image_idx]
        else:
            image_id = self.image_ids[idx]
        
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
            # Generate base mask from COCO annotations
            base_mask_pil = self._make_mask_from_annotations(image_id, W, H)
            
            if self.random_masks:
                # Generate random mask with padding and additional shapes
                mask_pil = self._generate_random_mask(base_mask_pil, W, H)
            else:
                # Use original vessel mask
                mask_pil = base_mask_pil
            
            mask_np = np.array(mask_pil, dtype=np.float32) / 255.0

        if self.patch_mode:
            # Extract random patch instead of resizing
            img_patch, mask_patch = self._extract_random_patch(img, mask_np, self.image_size)
            img_norm = (img_patch.astype(np.float32) / 255.0) * 2.0 - 1.0
        else:
            # Original behavior: resize
            img  = cv2.resize(img,     (self.image_size, self.image_size),
                              interpolation=cv2.INTER_LINEAR)
            mask_patch = cv2.resize(mask_np, (self.image_size, self.image_size),
                              interpolation=cv2.INTER_NEAREST)
            img_norm = (img.astype(np.float32) / 255.0) * 2.0 - 1.0

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
    parser.add_argument('--patch_mode', action='store_true',
                        help='Extract random patches instead of resizing entire image')
    parser.add_argument('--patches_per_image', type=int, default=4,
                        help='Number of patches to extract per image when using patch_mode')
    parser.add_argument('--foreground_prob', type=float, default=0.75,
                        help='Probability of biasing patch sampling toward foreground (mask) pixels')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # ---- Datasets ----
    train_dataset = ArcadeDataset(args.train_img, args.train_ann, args.input_size,
                                  mask_dir=args.train_mask, random_masks=args.random_masks,
                                  mask_padding=args.mask_padding, patch_mode=args.patch_mode,
                                  patches_per_image=args.patches_per_image,
                                  foreground_prob=args.foreground_prob)
    val_dataset   = ArcadeDataset(args.val_img,   args.val_ann,   args.input_size,
                                  mask_dir=args.val_mask, patch_mode=args.patch_mode,
                                  patches_per_image=args.patches_per_image,
                                  foreground_prob=args.foreground_prob)

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

    if args.patch_mode:
        base_train_imgs = len(train_dataset.image_ids)
        base_val_imgs = len(val_dataset.image_ids)
        print(f"Patch mode enabled: {args.patches_per_image} patches per image")
        print(f"Train: {base_train_imgs} images → {len(train_dataset)} patches | Val: {base_val_imgs} images → {len(val_dataset)} patches")
    else:
        print(f"Train: {len(train_dataset)} images | Val: {len(val_dataset)} images")

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
        f.write('epoch,train_loss,val_psnr,val_ssim,val_wasserstein,val_rmse\n')
    with open(analysis_log_path, 'w') as f:
        f.write('epoch,train_loss,l1_loss,ssim_loss,loss_change,psnr_realistic,learning_pattern\n')
    
    if drive_log_path:
        with open(drive_log_path, 'w') as f:
            f.write('epoch,train_loss,val_psnr,val_ssim,val_wasserstein,val_rmse\n')
    
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
        val_psnr = 0.0
        val_ssim = 0.0
        val_wasserstein = 0.0
        val_rmse = 0.0
        with torch.no_grad():
            for img, mask in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [val]"):
                img, mask = img.to(device), mask.to(device)
                output = model(img, mask)
                output = torch.clip(output, -1.0, 1.0)
                out_np = (output[:, 0].cpu().numpy() * 0.5 + 0.5) * 255.0
                gt_np  = (img[:, 0].cpu().numpy()    * 0.5 + 0.5) * 255.0
                for o, g in zip(out_np, gt_np):
                    val_psnr += psnr(o, g)
                    val_ssim += ssim_fn(o, g, data_range=255.0)
                    val_wasserstein += wasserstein_distance_2d(o, g)
                    val_rmse += rmse(o, g)
        val_psnr /= len(val_dataset)
        val_ssim /= len(val_dataset)
        val_wasserstein /= len(val_dataset)
        val_rmse /= len(val_dataset)

        scheduler.step()
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_psnr={val_psnr:.2f} dB | val_ssim={val_ssim:.4f} | val_wasserstein={val_wasserstein:.2f} | val_rmse={val_rmse:.2f}")

        with open(log_path, 'a') as f:
            f.write(f"{epoch},{train_loss:.4f},{val_psnr:.2f},{val_ssim:.4f},{val_wasserstein:.4f},{val_rmse:.4f}\n")
        if drive_log_path:
            with open(drive_log_path, 'a') as f:
                f.write(f"{epoch},{train_loss:.4f},{val_psnr:.2f},{val_ssim:.4f},{val_wasserstein:.4f},{val_rmse:.4f}\n")

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
            f.write(f"{epoch},{train_loss:.6f},{avg_l1_loss:.6f},{avg_ssim_loss:.6f},{loss_change:.6f},{psnr_realistic},{learning_pattern}\n")
        
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


if __name__ == '__main__':
    main()
