import torch
import numpy as np
try:
    from scipy.stats import wasserstein_distance
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from skimage.metrics import structural_similarity as ssim
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

def _load(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
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


def calculate_psnr(img1, img2, max_val=255.0):
    """Calculate PSNR between two images"""
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0  # Perfect match
    
    # Adjust max_val based on data range
    if np.max(img1) <= 1.0 and np.min(img1) >= -1.0:
        max_val = 2.0  # For [-1, 1] range
    elif np.max(img1) <= 1.0 and np.min(img1) >= 0.0:
        max_val = 1.0  # For [0, 1] range
    
    psnr = 20 * np.log10(max_val / np.sqrt(mse))
    return float(psnr)


def calculate_ssim(img1, img2, data_range=None):
    """Calculate SSIM between two images"""
    if not SKIMAGE_AVAILABLE:
        return 0.0  # Fallback if skimage not available
    
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    
    # Determine data range
    if data_range is None:
        if np.max(img1) <= 1.0 and np.min(img1) >= -1.0:
            data_range = 2.0  # For [-1, 1] range
        elif np.max(img1) <= 1.0 and np.min(img1) >= 0.0:
            data_range = 1.0  # For [0, 1] range
        else:
            data_range = 255.0  # For [0, 255] range
    
    ssim_value = ssim(img1, img2, data_range=data_range, channel_axis=None)
    return float(ssim_value)


def calculate_wasserstein(img1, img2):
    """Calculate Wasserstein distance between two image distributions"""
    if not SCIPY_AVAILABLE:
        return 0.0  # Fallback if scipy not available
    
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    
    # Flatten images to 1D distributions
    dist1 = img1.flatten()
    dist2 = img2.flatten()
    
    wd = wasserstein_distance(dist1, dist2)
    return float(wd)


def calculate_rmse(img1, img2):
    """Calculate RMSE between two images"""
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    
    mse = np.mean((img1 - img2) ** 2)
    rmse = np.sqrt(mse)
    return float(rmse)

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

