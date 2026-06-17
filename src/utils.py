import torch
import numpy as np
try:
    from scipy.stats import wasserstein_distance
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def _load(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device), weights_only=True)
    return checkpoint

def load_checkpoint(path, model, device, optimizer=None, reset_optimizer=True, is_dis=False):
    print("Load checkpoint from: {}".format(path))
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
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    return model


def save_checkpoint(model, optimizer, epoch, path, metrics=None):
    """Save model checkpoint with metrics"""
    checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer else None,
        'epoch': epoch
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
                print(f"  Rotated out: {os.path.basename(old_ckpt)}")
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

