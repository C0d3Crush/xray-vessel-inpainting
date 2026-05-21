import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil
from PIL import Image
import json


@pytest.fixture(scope="session")
def device():
    """Fixture providing CPU device for testing"""
    return torch.device("cpu")


@pytest.fixture(scope="session") 
def temp_dir():
    """Fixture providing temporary directory for test files"""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_image():
    """Create a sample grayscale image tensor"""
    return torch.randn(1, 1, 256, 256)


@pytest.fixture
def sample_mask():
    """Create a sample binary mask tensor"""
    mask = torch.zeros(1, 1, 256, 256)
    mask[:, :, 50:200, 50:200] = 1.0
    return mask


@pytest.fixture
def sample_batch():
    """Create a sample batch of images and masks"""
    images = torch.randn(4, 1, 64, 64)
    masks = torch.zeros(4, 1, 64, 64)
    masks[:, :, 16:48, 16:48] = 1.0
    return {"images": images, "masks": masks}


@pytest.fixture
def mock_coco_annotation():
    """Create mock COCO annotation for testing"""
    return {
        "images": [
            {
                "id": 1,
                "file_name": "test_image.png",
                "width": 256,
                "height": 256
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "segmentation": [
                    [50, 50, 200, 50, 200, 200, 50, 200]
                ],
                "area": 22500,
                "bbox": [50, 50, 150, 150]
            }
        ],
        "categories": [
            {
                "id": 1,
                "name": "vessel",
                "supercategory": "anatomy"
            }
        ]
    }


@pytest.fixture
def mock_dataset_files(temp_dir, mock_coco_annotation):
    """Create mock dataset files in temporary directory"""
    # Create annotation file
    ann_file = temp_dir / "annotations.json"
    with open(ann_file, 'w') as f:
        json.dump(mock_coco_annotation, f)
    
    # Create image directory and sample image
    img_dir = temp_dir / "images"
    img_dir.mkdir()
    
    img_file = img_dir / "test_image.png" 
    img_array = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
    Image.fromarray(img_array, mode='L').save(img_file)
    
    return {
        "annotation_file": ann_file,
        "image_dir": img_dir,
        "image_file": img_file
    }


@pytest.fixture
def checkpoint_dir(temp_dir):
    """Create temporary checkpoint directory"""
    ckpt_dir = temp_dir / "checkpoints"
    ckpt_dir.mkdir()
    return ckpt_dir


@pytest.fixture
def sample_checkpoint(checkpoint_dir, device):
    """Create a sample model checkpoint"""
    from src.network.network_pro import Inpaint
    
    model = Inpaint(input_size=64, device=device)
    checkpoint = {
        'epoch': 10,
        'model_state_dict': model.state_dict(),
        'train_loss': 0.5,
        'val_psnr': 25.0,
        'val_ssim': 0.8,
        'val_wasserstein': 0.1,
        'val_rmse': 0.05
    }
    
    ckpt_file = checkpoint_dir / "test_checkpoint.pth"
    torch.save(checkpoint, ckpt_file)
    return ckpt_file


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set random seeds for reproducible tests"""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)


@pytest.fixture
def mock_training_log(temp_dir):
    """Create mock training log CSV"""
    import pandas as pd
    
    data = {
        'epoch': [1, 2, 3, 4, 5],
        'train_loss': [1.0, 0.8, 0.6, 0.5, 0.4],
        'val_psnr': [20.0, 22.0, 24.0, 25.0, 26.0],
        'val_ssim': [0.5, 0.6, 0.7, 0.75, 0.8],
        'val_wasserstein': [0.5, 0.4, 0.3, 0.2, 0.1],
        'val_rmse': [0.2, 0.15, 0.1, 0.08, 0.05]
    }
    
    df = pd.DataFrame(data)
    log_file = temp_dir / "training_log.csv"
    df.to_csv(log_file, index=False)
    return log_file