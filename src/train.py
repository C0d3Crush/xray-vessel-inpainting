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

    def __init__(self, img_dir, ann_path, image_size=256, mask_dir=None, random_masks=False, mask_padding=10, patch_mode=False, patches_per_image=4):
        self.img_dir      = img_dir
        self.image_size   = image_size
        self.mask_dir     = mask_dir
        self.random_masks = random_masks
        self.mask_padding = mask_padding
        self.patch_mode   = patch_mode
        self.patches_per_image = patches_per_image

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

    def __len__(self):
        if self.patch_mode:
            return len(self.image_ids) * self.patches_per_image
        return len(self.image_ids)

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

    def _extract_random_patch(self, img, mask, patch_size):
        """Extract a random patch from the image and mask."""
        H, W = img.shape
        
        # Ensure patch fits within image
        if H < patch_size or W < patch_size:
            # Pad image if smaller than patch size
            pad_h = max(0, patch_size - H)
            pad_w = max(0, patch_size - W)
            img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
            H, W = img.shape
        
        # Random top-left corner
        max_y = H - patch_size
        max_x = W - patch_size
        y = np.random.randint(0, max_y + 1)
        x = np.random.randint(0, max_x + 1)
        
        # Extract patch
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

        # Load image
        img_path = os.path.join(self.img_dir, info['file_name'])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Load or generate mask
        if self.mask_dir is not None:
            mask_path = os.path.join(self.mask_dir, info['file_name'])
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
        return loss_mask * self.mask_weight + loss_valid * self.valid_weight + self.ssim_weight * loss_ssim


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
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # ---- Datasets ----
    train_dataset = ArcadeDataset(args.train_img, args.train_ann, args.input_size,
                                  mask_dir=args.train_mask, random_masks=args.random_masks,
                                  mask_padding=args.mask_padding, patch_mode=args.patch_mode,
                                  patches_per_image=args.patches_per_image)
    val_dataset   = ArcadeDataset(args.val_img,   args.val_ann,   args.input_size,
                                  mask_dir=args.val_mask, patch_mode=args.patch_mode,
                                  patches_per_image=args.patches_per_image)

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
    with open(log_path, 'w') as f:
        f.write('epoch,train_loss,val_psnr,val_ssim,val_wasserstein,val_rmse\n')
    if drive_log_path:
        with open(drive_log_path, 'w') as f:
            f.write('epoch,train_loss,val_psnr,val_ssim,val_wasserstein,val_rmse\n')

    for epoch in range(1, args.epochs + 1):
        # -- Train --
        model.train()
        train_loss = 0.0
        prog = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]")
        for img, mask in prog:
            img, mask = img.to(device), mask.to(device)
            optimizer.zero_grad()
            output = model(img, mask)
            loss   = criterion(output, img, mask)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            prog.set_postfix(loss=f"{loss.item():.4f}")
        train_loss /= len(train_loader)

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
