import pytest
import torch
import json
import numpy as np
from pathlib import Path
from PIL import Image
from unittest.mock import patch, MagicMock


class TestArcadeDataset:
    """Test cases for ArcadeDataset class"""
    
    @pytest.fixture
    def arcade_dataset(self, mock_dataset_files):
        """Create ArcadeDataset instance with mock data"""
        from src.train import ArcadeDataset
        
        return ArcadeDataset(
            img_path=str(mock_dataset_files["image_dir"]),
            ann_path=str(mock_dataset_files["annotation_file"]),
            input_size=64,
            max_dataset_size=None
        )
    
    def test_dataset_initialization(self, arcade_dataset, mock_dataset_files):
        """Test dataset initialization"""
        assert arcade_dataset.input_size == 64
        assert arcade_dataset.img_path == str(mock_dataset_files["image_dir"])
        assert arcade_dataset.ann_path == str(mock_dataset_files["annotation_file"])
        assert len(arcade_dataset.coco.imgs) == 1
        assert len(arcade_dataset.image_ids) == 1
    
    def test_dataset_length(self, arcade_dataset):
        """Test dataset length"""
        assert len(arcade_dataset) == 1
    
    def test_dataset_getitem(self, arcade_dataset):
        """Test dataset __getitem__ method"""
        item = arcade_dataset[0]
        
        assert "image" in item
        assert "mask" in item
        assert "image_id" in item
        
        # Check tensor shapes and types
        assert isinstance(item["image"], torch.Tensor)
        assert isinstance(item["mask"], torch.Tensor) 
        assert item["image"].shape == (1, 64, 64)  # Grayscale, resized
        assert item["mask"].shape == (1, 64, 64)
        
        # Check value ranges
        assert torch.all(item["image"] >= -1.0) and torch.all(item["image"] <= 1.0)
        assert torch.all(item["mask"] >= 0.0) and torch.all(item["mask"] <= 1.0)
        assert torch.all((item["mask"] == 0.0) | (item["mask"] == 1.0))  # Binary mask
    
    def test_stenosis_exclusion(self, temp_dir):
        """Test that stenosis category (ID 26) is excluded from masks"""
        from src.train import ArcadeDataset
        
        # Create annotation with stenosis category
        annotation_with_stenosis = {
            "images": [
                {"id": 1, "file_name": "test.png", "width": 256, "height": 256}
            ],
            "annotations": [
                {
                    "id": 1, "image_id": 1, "category_id": 1,
                    "segmentation": [[50, 50, 100, 50, 100, 100, 50, 100]],
                    "area": 2500, "bbox": [50, 50, 50, 50]
                },
                {
                    "id": 2, "image_id": 1, "category_id": 26,  # Stenosis category
                    "segmentation": [[150, 150, 200, 150, 200, 200, 150, 200]],
                    "area": 2500, "bbox": [150, 150, 50, 50]
                }
            ],
            "categories": [
                {"id": 1, "name": "vessel"},
                {"id": 26, "name": "stenosis"}
            ]
        }
        
        # Save annotation and create test image
        ann_file = temp_dir / "stenosis_test.json"
        with open(ann_file, 'w') as f:
            json.dump(annotation_with_stenosis, f)
        
        img_dir = temp_dir / "stenosis_images"
        img_dir.mkdir()
        img_file = img_dir / "test.png"
        img_array = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        Image.fromarray(img_array, mode='L').save(img_file)
        
        # Create dataset and check mask
        dataset = ArcadeDataset(str(img_dir), str(ann_file), input_size=256)
        item = dataset[0]
        
        # Mask should only contain vessel annotation, not stenosis
        assert torch.any(item["mask"] == 1.0)  # Should have some masked regions
        
        # Check that stenosis annotations are filtered out during loading
        anns = dataset.coco.getAnnIds(imgIds=[1])
        categories = [dataset.coco.anns[ann_id]['category_id'] for ann_id in anns]
        assert 26 not in categories  # Stenosis should be filtered out
    
    def test_mask_generation(self, arcade_dataset):
        """Test mask generation from COCO polygons"""
        item = arcade_dataset[0]
        mask = item["mask"]
        
        # Should have masked regions (vessel areas)
        assert torch.any(mask == 1.0)
        assert torch.any(mask == 0.0)
        
        # Mask should be binary
        unique_values = torch.unique(mask)
        assert len(unique_values) <= 2
        assert torch.all((unique_values == 0.0) | (unique_values == 1.0))
    
    def test_image_preprocessing(self, arcade_dataset):
        """Test image preprocessing (grayscale conversion and normalization)"""
        item = arcade_dataset[0]
        image = item["image"]
        
        # Check normalization to [-1, 1] range
        assert torch.all(image >= -1.0)
        assert torch.all(image <= 1.0)
        
        # Check single channel (grayscale)
        assert image.shape[0] == 1
    
    def test_resize_functionality(self, temp_dir):
        """Test dataset with different input sizes"""
        from src.train import ArcadeDataset
        
        # Create test data
        annotation = {
            "images": [{"id": 1, "file_name": "test.png", "width": 512, "height": 512}],
            "annotations": [{
                "id": 1, "image_id": 1, "category_id": 1,
                "segmentation": [[100, 100, 200, 100, 200, 200, 100, 200]],
                "area": 10000, "bbox": [100, 100, 100, 100]
            }],
            "categories": [{"id": 1, "name": "vessel"}]
        }
        
        ann_file = temp_dir / "resize_test.json"
        with open(ann_file, 'w') as f:
            json.dump(annotation, f)
        
        img_dir = temp_dir / "resize_images"
        img_dir.mkdir()
        img_file = img_dir / "test.png"
        img_array = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
        Image.fromarray(img_array, mode='L').save(img_file)
        
        # Test different input sizes
        for input_size in [64, 128, 256]:
            dataset = ArcadeDataset(str(img_dir), str(ann_file), input_size=input_size)
            item = dataset[0]
            
            assert item["image"].shape == (1, input_size, input_size)
            assert item["mask"].shape == (1, input_size, input_size)
    
    def test_max_dataset_size_limit(self, mock_dataset_files):
        """Test dataset size limiting"""
        from src.train import ArcadeDataset
        
        # Create dataset with size limit
        dataset = ArcadeDataset(
            img_path=str(mock_dataset_files["image_dir"]),
            ann_path=str(mock_dataset_files["annotation_file"]),
            input_size=64,
            max_dataset_size=0  # Should limit to 0 items
        )
        
        assert len(dataset) == 0
    
    @pytest.mark.parametrize("random_masks", [True, False])
    def test_random_masks_option(self, mock_dataset_files, random_masks):
        """Test random mask generation option"""
        from src.train import ArcadeDataset
        
        dataset = ArcadeDataset(
            img_path=str(mock_dataset_files["image_dir"]),
            ann_path=str(mock_dataset_files["annotation_file"]),
            input_size=64,
            random_masks=random_masks
        )
        
        item1 = dataset[0]
        item2 = dataset[0]
        
        if random_masks:
            # Masks should be different when random
            assert not torch.equal(item1["mask"], item2["mask"])
        else:
            # Masks should be same when using COCO annotations
            assert torch.equal(item1["mask"], item2["mask"])


class TestDatasetUtils:
    """Test dataset utility functions"""
    
    def test_polygon_to_mask_conversion(self):
        """Test conversion of polygon to binary mask"""
        # This tests the internal polygon rasterization functionality
        from pycocotools import mask as maskUtils
        import numpy as np
        
        # Simple rectangle polygon
        polygon = [50, 50, 150, 50, 150, 150, 50, 150]
        
        # Create RLE mask
        rle = maskUtils.frPyObjects([polygon], 200, 200)
        binary_mask = maskUtils.decode(rle[0])
        
        # Check mask properties
        assert binary_mask.shape == (200, 200)
        assert binary_mask.dtype == np.uint8
        assert np.any(binary_mask == 1)
        assert np.any(binary_mask == 0)
        
        # Check that mask covers expected region
        assert binary_mask[100, 100] == 1  # Inside polygon
        assert binary_mask[25, 25] == 0    # Outside polygon
    
    def test_image_normalization(self):
        """Test image normalization function"""
        # Create test image
        img_array = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        img_pil = Image.fromarray(img_array, mode='L')
        
        # Convert to tensor and normalize (simulating dataset preprocessing)
        import torchvision.transforms as transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Maps [0,1] to [-1,1]
        ])
        
        normalized = transform(img_pil)
        
        # Check normalization
        assert normalized.shape == (1, 256, 256)
        assert torch.all(normalized >= -1.0)
        assert torch.all(normalized <= 1.0)
    
    def test_coco_annotation_loading(self, mock_coco_annotation, temp_dir):
        """Test COCO annotation loading"""
        from pycocotools.coco import COCO
        
        # Save annotation to file
        ann_file = temp_dir / "test_coco.json"
        with open(ann_file, 'w') as f:
            json.dump(mock_coco_annotation, f)
        
        # Load with COCO API
        coco = COCO(str(ann_file))
        
        # Check loaded data
        assert len(coco.imgs) == 1
        assert len(coco.anns) == 1
        assert len(coco.cats) == 1
        
        # Check image info
        img_info = coco.imgs[1]
        assert img_info['file_name'] == 'test_image.png'
        assert img_info['width'] == 256
        assert img_info['height'] == 256
        
        # Check annotation info
        ann_info = coco.anns[1]
        assert ann_info['category_id'] == 1
        assert ann_info['image_id'] == 1
        assert len(ann_info['segmentation'][0]) == 8  # 4 points * 2 coordinates


class TestDataLoading:
    """Test data loading with PyTorch DataLoader"""
    
    def test_dataloader_integration(self, arcade_dataset):
        """Test dataset works with PyTorch DataLoader"""
        from torch.utils.data import DataLoader
        
        dataloader = DataLoader(
            arcade_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0  # Avoid multiprocessing issues in tests
        )
        
        batch = next(iter(dataloader))
        
        assert "image" in batch
        assert "mask" in batch
        assert "image_id" in batch
        
        assert batch["image"].shape == (1, 1, 64, 64)  # [batch, channel, h, w]
        assert batch["mask"].shape == (1, 1, 64, 64)
    
    def test_batch_collation(self, mock_dataset_files):
        """Test batching multiple samples"""
        from src.train import ArcadeDataset
        from torch.utils.data import DataLoader
        
        # Create dataset with multiple identical images for testing
        dataset = ArcadeDataset(
            img_path=str(mock_dataset_files["image_dir"]),
            ann_path=str(mock_dataset_files["annotation_file"]),
            input_size=64
        )
        
        # Simulate multiple samples by repeating
        extended_dataset = torch.utils.data.ConcatDataset([dataset] * 4)
        
        dataloader = DataLoader(extended_dataset, batch_size=2, shuffle=False)
        batch = next(iter(dataloader))
        
        assert batch["image"].shape == (2, 1, 64, 64)
        assert batch["mask"].shape == (2, 1, 64, 64)
        assert len(batch["image_id"]) == 2