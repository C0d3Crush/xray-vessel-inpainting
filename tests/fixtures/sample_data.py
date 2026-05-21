"""
Sample data fixtures for testing
"""

import torch
import numpy as np
from pathlib import Path
import json
from PIL import Image
import tempfile


def create_sample_coco_annotation(num_images=3, num_annotations=3):
    """Create a sample COCO annotation for testing"""
    annotation = {
        "images": [
            {
                "id": i+1,
                "file_name": f"image_{i:03d}.png",
                "width": 256,
                "height": 256
            }
            for i in range(num_images)
        ],
        "annotations": [
            {
                "id": i+1,
                "image_id": (i % num_images) + 1,
                "category_id": 1,
                "segmentation": [
                    [50 + i*10, 50, 100 + i*10, 50, 100 + i*10, 100, 50 + i*10, 100]
                ],
                "area": 2500,
                "bbox": [50 + i*10, 50, 50, 50]
            }
            for i in range(num_annotations)
        ],
        "categories": [
            {"id": 1, "name": "vessel", "supercategory": "anatomy"}
        ]
    }
    
    # Add stenosis category for testing exclusion
    annotation["categories"].append({
        "id": 26, "name": "stenosis", "supercategory": "pathology"
    })
    
    return annotation


def create_sample_images(output_dir, num_images=3, size=(256, 256)):
    """Create sample grayscale images for testing"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    image_paths = []
    for i in range(num_images):
        # Create varied test images
        img_array = np.random.randint(50 + i*20, 200 - i*10, size, dtype=np.uint8)
        img = Image.fromarray(img_array, mode='L')
        
        img_path = output_path / f"image_{i:03d}.png"
        img.save(img_path)
        image_paths.append(img_path)
    
    return image_paths


def create_sample_masks(output_dir, num_masks=3, size=(256, 256)):
    """Create sample binary masks for testing"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    mask_paths = []
    for i in range(num_masks):
        # Create varied mask shapes
        mask_array = np.zeros(size, dtype=np.uint8)
        
        # Different mask patterns for variety
        if i % 3 == 0:
            # Square mask
            start = 64 + i*10
            end = 192 - i*10
            mask_array[start:end, start:end] = 255
        elif i % 3 == 1:
            # Circular mask
            center = (size[0]//2, size[1]//2)
            radius = 50 + i*5
            y, x = np.ogrid[:size[0], :size[1]]
            mask = (x - center[1])**2 + (y - center[0])**2 <= radius**2
            mask_array[mask] = 255
        else:
            # Irregular mask
            mask_array[50+i*5:150+i*5, 100-i*5:200-i*5] = 255
        
        mask = Image.fromarray(mask_array, mode='L')
        mask_path = output_path / f"mask_{i:03d}.png"
        mask.save(mask_path)
        mask_paths.append(mask_path)
    
    return mask_paths


def create_test_dataset(base_dir, num_samples=3):
    """Create a complete test dataset with images, masks, and annotations"""
    base_path = Path(base_dir)
    
    # Create directory structure
    dirs = {
        'train_img': base_path / 'train' / 'images',
        'train_ann': base_path / 'train' / 'annotations',
        'val_img': base_path / 'val' / 'images',
        'val_ann': base_path / 'val' / 'annotations'
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create train images and annotations
    train_images = create_sample_images(dirs['train_img'], num_samples)
    train_annotation = create_sample_coco_annotation(num_samples, num_samples)
    
    train_ann_file = dirs['train_ann'] / 'train.json'
    with open(train_ann_file, 'w') as f:
        json.dump(train_annotation, f)
    
    # Create val images and annotations  
    val_images = create_sample_images(dirs['val_img'], num_samples)
    val_annotation = create_sample_coco_annotation(num_samples, num_samples)
    
    # Update filenames for val set
    for i, img_info in enumerate(val_annotation['images']):
        img_info['file_name'] = f'image_{i:03d}.png'
    
    val_ann_file = dirs['val_ann'] / 'val.json'
    with open(val_ann_file, 'w') as f:
        json.dump(val_annotation, f)
    
    return {
        'train_img': str(dirs['train_img']),
        'train_ann': str(train_ann_file),
        'val_img': str(dirs['val_img']),
        'val_ann': str(val_ann_file)
    }


def create_mock_checkpoint(checkpoint_path, input_size=64):
    """Create a mock model checkpoint for testing"""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path.cwd() / "src"))
    
    from src.network.network_pro import Inpaint
    
    device = torch.device('cpu')
    model = Inpaint(input_size=input_size, device=device)
    
    checkpoint = {
        'epoch': 10,
        'model_state_dict': model.state_dict(),
        'train_loss': 0.45,
        'val_psnr': 28.5,
        'val_ssim': 0.82,
        'val_wasserstein': 0.15,
        'val_rmse': 0.08,
        'input_size': input_size
    }
    
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def create_training_log_csv(csv_path, num_epochs=10):
    """Create a mock training log CSV file"""
    import pandas as pd
    
    # Generate realistic training progression
    epochs = list(range(1, num_epochs + 1))
    
    # Decreasing loss with some noise
    train_loss = [1.2 - 0.08 * i + np.random.normal(0, 0.05) for i in epochs]
    train_loss = np.clip(train_loss, 0.1, 2.0)  # Reasonable bounds
    
    # Increasing PSNR with some noise
    val_psnr = [18 + 1.2 * i + np.random.normal(0, 1.0) for i in epochs]
    val_psnr = np.clip(val_psnr, 15, 35)
    
    # Increasing SSIM
    val_ssim = [0.4 + 0.04 * i + np.random.normal(0, 0.02) for i in epochs]
    val_ssim = np.clip(val_ssim, 0.3, 0.95)
    
    # Decreasing Wasserstein distance
    val_wasserstein = [0.8 - 0.06 * i + np.random.normal(0, 0.03) for i in epochs]
    val_wasserstein = np.clip(val_wasserstein, 0.05, 1.0)
    
    # Decreasing RMSE
    val_rmse = [0.5 - 0.03 * i + np.random.normal(0, 0.01) for i in epochs]
    val_rmse = np.clip(val_rmse, 0.02, 0.6)
    
    data = {
        'epoch': epochs,
        'train_loss': train_loss,
        'val_psnr': val_psnr,
        'val_ssim': val_ssim,
        'val_wasserstein': val_wasserstein,
        'val_rmse': val_rmse
    }
    
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    return csv_path