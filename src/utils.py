import json
import logging
import os
import subprocess
import torch
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)

try:
    from scipy.stats import wasserstein_distance
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

STENOSIS_CATEGORY_ID = 26


def load_coco_annotations(ann_path):
    """
    Parse a COCO JSON file and return annotation lookups.

    Stenosis annotations (category 26) are excluded.

    Returns:
        id_to_info (dict): image_id -> image info dict
        anns_by_image (defaultdict): image_id -> list of annotations
        image_ids (list): image IDs that have at least one annotation
    """
    with open(ann_path) as f:
        coco = json.load(f)

    id_to_info = {img['id']: img for img in coco['images']}
    anns_by_image = defaultdict(list)

    for ann in coco['annotations']:
        if ann['category_id'] != STENOSIS_CATEGORY_ID:
            anns_by_image[ann['image_id']].append(ann)

    image_ids = [img_id for img_id in id_to_info if anns_by_image[img_id]]
    return id_to_info, anns_by_image, image_ids


def _load(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device), weights_only=True)
    return checkpoint

def load_checkpoint(path, model, device, optimizer=None, reset_optimizer=True, is_dis=False):
    logger.info("Load checkpoint from: {}".format(path))
    checkpoint = _load(path, device)
    if is_dis:
        s = checkpoint["disc"]
    else:
        s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s, strict=True)
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            logger.info("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    return model


def _git_hash():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        ).decode().strip()
    except Exception:
        return 'unknown'


def save_checkpoint(model, optimizer, epoch, path, metrics=None):
    """Save model checkpoint with metrics"""
    checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer else None,
        'epoch': epoch,
        'git_hash': _git_hash(),
    }
    if metrics:
        checkpoint.update(metrics)
    torch.save(checkpoint, path)
    return checkpoint


def rotate_checkpoints(output_dir, keep_checkpoints):
    """Keep only the most recent N checkpoint files"""
    if keep_checkpoints <= 0:
        return
    
    import glob
    import os
    
    # Find all epoch checkpoints (exclude best.pth)
    checkpoints = glob.glob(os.path.join(output_dir, 'epoch_*.pth'))
    
    if len(checkpoints) > keep_checkpoints:
        # Sort by modification time (newest first)
        checkpoints.sort(key=os.path.getmtime, reverse=True)
        
        # Remove old checkpoints
        for old_ckpt in checkpoints[keep_checkpoints:]:
            try:
                os.remove(old_ckpt)
                logger.debug(f"Rotated out: {os.path.basename(old_ckpt)}")
            except OSError:
                pass


def psnr(img1, img2):
    mse = np.mean((img1-img2)** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def rmse(img1, img2):
    """Root Mean Square Error between two images."""
    return np.sqrt(np.mean((img1 - img2) ** 2))

def wasserstein_distance_2d_batch(imgs1, imgs2):
    """Wasserstein distance for a batch. imgs1, imgs2: (B, H, W) in [0, 255]. Returns (B,) array."""
    if not SCIPY_AVAILABLE:
        flat1 = imgs1.reshape(len(imgs1), -1)
        flat2 = imgs2.reshape(len(imgs2), -1)
        return np.mean(np.abs(np.sort(flat1, axis=1) - np.sort(flat2, axis=1)), axis=1)
    return np.array([wasserstein_distance(imgs1[i].flatten(), imgs2[i].flatten()) for i in range(len(imgs1))])


def calculate_kl_divergence_batch(imgs1, imgs2, bins=256, epsilon=1e-10):
    """KL divergence for a batch. imgs1, imgs2: (B, H, W). Returns (B,) array."""
    B = imgs1.shape[0]
    flat1 = imgs1.reshape(B, -1).astype(np.float64)
    flat2 = imgs2.reshape(B, -1).astype(np.float64)
    mn1 = flat1.min(axis=1, keepdims=True)
    mx1 = flat1.max(axis=1, keepdims=True)
    mn2 = flat2.min(axis=1, keepdims=True)
    mx2 = flat2.max(axis=1, keepdims=True)
    flat1 = (flat1 - mn1) / (mx1 - mn1 + epsilon)
    flat2 = (flat2 - mn2) / (mx2 - mn2 + epsilon)
    hist1 = np.array([np.histogram(flat1[i], bins=bins, range=(0, 1), density=True)[0] for i in range(B)])
    hist2 = np.array([np.histogram(flat2[i], bins=bins, range=(0, 1), density=True)[0] for i in range(B)])
    hist1 = hist1 / hist1.sum(axis=1, keepdims=True) + epsilon
    hist2 = hist2 / hist2.sum(axis=1, keepdims=True) + epsilon
    return np.sum(hist1 * np.log(hist1 / hist2), axis=1)


def wasserstein_distance_2d(img1, img2):
    """
    Approximate 2D Wasserstein distance for grayscale images.
    Uses 1D Wasserstein on flattened pixel distributions for efficiency.
    
    Args:
        img1, img2: numpy arrays of shape (H, W) with pixel values [0, 255]
    
    Returns:
        float: Wasserstein distance (lower is better)
    """
    if not SCIPY_AVAILABLE:
        # Fallback to simpler Earth Mover's Distance approximation
        return np.mean(np.abs(np.sort(img1.flatten()) - np.sort(img2.flatten())))
    
    # Use scipy's 1D Wasserstein on flattened distributions
    return wasserstein_distance(img1.flatten(), img2.flatten())


def calculate_kl_divergence(img1, img2, bins=256, epsilon=1e-10):
    """
    Calculate Kullback-Leibler divergence between two images using histograms.
    
    KL divergence measures how different two probability distributions are.
    For medical inpainting, this shows how well the generated tissue matches 
    the original intensity distribution.
    
    Args:
        img1, img2: Input images (numpy arrays or torch tensors)
        bins: Number of histogram bins (default 256 for 8-bit images)
        epsilon: Small value to avoid log(0)
    
    Returns:
        float: KL divergence D_KL(P||Q) where P=img1, Q=img2 (lower is better)
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    
    # Normalize images to [0, 1] range for consistent histograms
    img1_norm = (img1 - img1.min()) / (img1.max() - img1.min() + epsilon)
    img2_norm = (img2 - img2.min()) / (img2.max() - img2.min() + epsilon)
    
    # Calculate histograms (probability distributions)
    hist1, _ = np.histogram(img1_norm.flatten(), bins=bins, range=(0, 1), density=True)
    hist2, _ = np.histogram(img2_norm.flatten(), bins=bins, range=(0, 1), density=True)
    
    # Normalize to probability distributions
    hist1 = hist1 / np.sum(hist1)
    hist2 = hist2 / np.sum(hist2)
    
    # Add epsilon to avoid log(0)
    hist1 = hist1 + epsilon
    hist2 = hist2 + epsilon
    
    # Calculate KL divergence: D_KL(P||Q) = sum(P * log(P/Q))
    kl_div = np.sum(hist1 * np.log(hist1 / hist2))
    
    return float(kl_div)

