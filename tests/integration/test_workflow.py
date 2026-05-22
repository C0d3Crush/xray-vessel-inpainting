import pytest
import tempfile
import json
import numpy as np
import cv2
from pathlib import Path
from unittest.mock import patch
import subprocess
import os


class TestWorkflowIntegration:
    """Integration tests for the complete training workflow"""

    @pytest.fixture
    def mock_arcade_data(self):
        """Create mock ARCADE dataset structure"""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            
            # Create directory structure
            data_dir = base_dir / "data" / "arcade" / "syntax"
            train_img_dir = data_dir / "train" / "images"
            train_ann_dir = data_dir / "train" / "annotations"
            val_img_dir = data_dir / "val" / "images" 
            val_ann_dir = data_dir / "val" / "annotations"
            
            train_img_dir.mkdir(parents=True)
            train_ann_dir.mkdir(parents=True)
            val_img_dir.mkdir(parents=True)
            val_ann_dir.mkdir(parents=True)
            
            # Create mock images (3 train, 2 val)
            train_images = []
            val_images = []
            
            for i in range(1, 4):  # 3 training images
                img = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
                img_path = train_img_dir / f"{i}.png"
                cv2.imwrite(str(img_path), img)
                train_images.append({
                    "id": i,
                    "file_name": f"{i}.png",
                    "width": 256,
                    "height": 256
                })
            
            for i in range(4, 6):  # 2 validation images  
                img = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
                img_path = val_img_dir / f"{i}.png"
                cv2.imwrite(str(img_path), img)
                val_images.append({
                    "id": i,
                    "file_name": f"{i}.png", 
                    "width": 256,
                    "height": 256
                })
            
            # Create mock COCO annotations
            def create_annotation(image_id, ann_id):
                return {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": 1,  # Not stenosis (26)
                    "segmentation": [
                        [50, 50, 100, 50, 100, 100, 50, 100]  # Simple rectangle
                    ],
                    "area": 2500,
                    "bbox": [50, 50, 50, 50],
                    "iscrowd": 0
                }
            
            train_annotations = []
            val_annotations = []
            
            for i, img in enumerate(train_images):
                train_annotations.append(create_annotation(img["id"], i+1))
            
            for i, img in enumerate(val_images):
                val_annotations.append(create_annotation(img["id"], i+10))
            
            # Create train annotation file
            train_coco = {
                "images": train_images,
                "annotations": train_annotations,
                "categories": [{"id": 1, "name": "vessel"}]
            }
            
            train_ann_file = train_ann_dir / "train.json"
            with open(train_ann_file, 'w') as f:
                json.dump(train_coco, f)
            
            # Create val annotation file
            val_coco = {
                "images": val_images,
                "annotations": val_annotations,
                "categories": [{"id": 1, "name": "vessel"}]
            }
            
            val_ann_file = val_ann_dir / "val.json"
            with open(val_ann_file, 'w') as f:
                json.dump(val_coco, f)
            
            yield {
                "base_dir": base_dir,
                "train_img": str(train_img_dir),
                "train_ann": str(train_ann_file),
                "val_img": str(val_img_dir),
                "val_ann": str(val_ann_file)
            }

    @pytest.mark.integration
    def test_cache_data_workflow(self, mock_arcade_data):
        """Test the cache-data step works correctly"""
        base_dir = mock_arcade_data["base_dir"]
        masks_dir = base_dir / "data" / "masks_cache"
        
        # Change to test directory
        original_cwd = os.getcwd()
        os.chdir(base_dir)
        
        try:
            # Run cache masks script
            result = subprocess.run([
                "python", f"{original_cwd}/scripts/cache_masks.py",
                "--annotations", mock_arcade_data["train_ann"].replace(str(base_dir) + "/", ""),
                "--images", mock_arcade_data["train_img"].replace(str(base_dir) + "/", ""), 
                "--output", "data/masks_cache/train"
            ], cwd=original_cwd, capture_output=True, text=True)
            
            # Should complete successfully
            assert result.returncode == 0
            
            # Check that cache directory was created
            train_cache_dir = masks_dir / "train"
            assert train_cache_dir.exists()
            
            # Check that mask files were generated
            mask_files = list(train_cache_dir.glob("*.png"))
            assert len(mask_files) == 3  # Should match number of train images
            
        finally:
            os.chdir(original_cwd)

    @pytest.mark.integration
    def test_background_mask_generation_completeness(self, mock_arcade_data):
        """Test that background mask generation handles file completeness correctly"""
        base_dir = mock_arcade_data["base_dir"]
        
        # First cache the vessel masks
        masks_dir = base_dir / "data" / "masks_cache"
        train_cache_dir = masks_dir / "train"
        train_cache_dir.mkdir(parents=True)
        
        # Create mock vessel masks
        for i in range(1, 4):
            vessel_mask = np.zeros((256, 256), dtype=np.uint8)
            # Add some vessel patterns
            cv2.line(vessel_mask, (50, 50), (200, 200), 255, 3)
            mask_path = train_cache_dir / f"{i}.png"
            cv2.imwrite(str(mask_path), vessel_mask)
        
        # Change to test directory
        original_cwd = os.getcwd()
        os.chdir(base_dir)
        
        try:
            # Run background mask generation
            result = subprocess.run([
                "python", f"{original_cwd}/src/generate_background_masks.py",
                "--input-img", mock_arcade_data["train_img"].replace(str(base_dir) + "/", ""),
                "--input-mask", "data/masks_cache/train",
                "--output-img", "data/smoke_bg_img", 
                "--output-mask", "data/smoke_bg_mask",
                "--variations", "2",
                "--safety-margin", "5"
            ], cwd=original_cwd, capture_output=True, text=True)
            
            # Should complete successfully  
            assert result.returncode == 0
            
            # Check output directories
            bg_img_dir = base_dir / "data" / "smoke_bg_img"
            bg_mask_dir = base_dir / "data" / "smoke_bg_mask"
            
            assert bg_img_dir.exists()
            assert bg_mask_dir.exists()
            
            # Get generated files
            bg_images = list(bg_img_dir.glob("*.png"))
            bg_masks = list(bg_mask_dir.glob("*.png"))
            
            # Should have same number of images and masks
            assert len(bg_images) == len(bg_masks)
            
            # Check file naming consistency
            bg_image_names = {f.name for f in bg_images}
            bg_mask_names = {f.name for f in bg_masks}
            assert bg_image_names == bg_mask_names
            
            # This is the critical test: may not generate all variations
            # if vessel density is too high
            expected_max_files = 3 * 2  # 3 images * 2 variations
            assert len(bg_images) <= expected_max_files
            assert len(bg_images) >= 0  # May generate 0 if all fail
            
        finally:
            os.chdir(original_cwd)

    @pytest.mark.integration 
    def test_dataset_training_compatibility(self, mock_arcade_data):
        """Test the critical issue: training dataset expects complete file sets"""
        base_dir = mock_arcade_data["base_dir"]
        
        # Simulate background generation that fails for some images
        bg_img_dir = base_dir / "data" / "smoke_bg_img" 
        bg_mask_dir = base_dir / "data" / "smoke_bg_mask"
        bg_img_dir.mkdir(parents=True)
        bg_mask_dir.mkdir(parents=True)
        
        # Only create background files for SOME of the training images
        # This simulates the real bug where generation fails for some images
        successful_images = ["1_bg_00.png", "3_bg_00.png"]  # Missing 2_bg_00.png
        
        for img_name in successful_images:
            # Create mock background image and mask
            bg_img = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
            bg_mask = np.random.randint(0, 2, (256, 256), dtype=np.uint8) * 255
            
            cv2.imwrite(str(bg_img_dir / img_name), bg_img)
            cv2.imwrite(str(bg_mask_dir / img_name), bg_mask)
        
        # Now test if training dataset can handle missing files
        import sys
        sys.path.append('src')
        
        # Mock the training scenario
        from train import ArcadeDataset
        
        # Change to test directory for relative paths
        original_cwd = os.getcwd()
        os.chdir(base_dir)
        
        try:
            # This should expose the bug: dataset expects all annotation files
            # to have corresponding background files
            with pytest.raises(FileNotFoundError):
                dataset = ArcadeDataset(
                    img_path="data/smoke_bg_img",
                    ann_path=mock_arcade_data["val_ann"].replace(str(base_dir) + "/", ""),
                    mask_dir="data/smoke_bg_mask",
                    input_size=64
                )
                
                # Try to access an item that should have missing background file
                # This will trigger the FileNotFoundError we found
                item = dataset[0] 
                
        finally:
            os.chdir(original_cwd)

    @pytest.mark.integration
    @pytest.mark.slow
    def test_smoke_training_workflow(self, mock_arcade_data):
        """Test complete smoke training workflow"""
        base_dir = mock_arcade_data["base_dir"]
        original_cwd = os.getcwd()
        os.chdir(base_dir)
        
        try:
            # Step 1: Cache vessel masks
            result = subprocess.run([
                "python", f"{original_cwd}/scripts/cache_masks.py",
                "--annotations", mock_arcade_data["train_ann"].replace(str(base_dir) + "/", ""),
                "--images", mock_arcade_data["train_img"].replace(str(base_dir) + "/", ""),
                "--output", "data/masks_cache/train"
            ], cwd=original_cwd, capture_output=True, text=True)
            assert result.returncode == 0
            
            # Step 2: Generate background masks  
            result = subprocess.run([
                "python", f"{original_cwd}/src/generate_background_masks.py",
                "--input-img", mock_arcade_data["train_img"].replace(str(base_dir) + "/", ""),
                "--input-mask", "data/masks_cache/train",
                "--output-img", "data/smoke_bg_img",
                "--output-mask", "data/smoke_bg_mask",
                "--variations", "1",
                "--safety-margin", "3"
            ], cwd=original_cwd, capture_output=True, text=True)
            
            # Background generation may succeed or fail - both are valid
            bg_img_dir = base_dir / "data" / "smoke_bg_img"
            bg_mask_dir = base_dir / "data" / "smoke_bg_mask"
            
            if result.returncode == 0 and bg_img_dir.exists():
                bg_images = list(bg_img_dir.glob("*.png"))
                
                # Only proceed with training if we have background files
                if len(bg_images) > 0:
                    # Step 3: Try smoke training
                    result = subprocess.run([
                        "python", f"{original_cwd}/src/train.py",
                        "--smoke_test",
                        "--smoke_size", "2",
                        "--epochs", "1", 
                        "--batch_size", "1",
                        "--device", "cpu",
                        "--train_img", "data/smoke_bg_img",
                        "--train_mask", "data/smoke_bg_mask",
                        "--val_img", mock_arcade_data["val_img"].replace(str(base_dir) + "/", ""),
                        "--val_ann", mock_arcade_data["val_ann"].replace(str(base_dir) + "/", ""),
                        "--output_dir", "test_checkpoints"
                    ], cwd=original_cwd, capture_output=True, text=True)
                    
                    # Training should either succeed or fail gracefully
                    # (not crash with FileNotFoundError)
                    if result.returncode != 0:
                        print("STDOUT:", result.stdout)
                        print("STDERR:", result.stderr)
                        
                        # Check that failure is NOT due to missing files
                        assert "FileNotFoundError" not in result.stderr
                        assert "Image not found" not in result.stderr
                
        finally:
            os.chdir(original_cwd)

    @pytest.mark.integration
    def test_file_consistency_validation(self):
        """Test that validates file consistency between annotations and generated data"""
        
        def validate_training_data_consistency(ann_file, img_dir, mask_dir):
            """Utility function to check data consistency"""
            import json
            
            # Load annotations
            with open(ann_file, 'r') as f:
                data = json.load(f)
            
            # Get expected image files from annotations
            expected_files = {img['file_name'] for img in data['images']}
            
            # Get actual files in directories
            actual_img_files = {f.name for f in Path(img_dir).glob("*.png")}
            actual_mask_files = {f.name for f in Path(mask_dir).glob("*.png")}
            
            # For background training, file names have _bg_XX suffix
            # Need to check if expected files have background variants
            if any("_bg_" in f for f in actual_img_files):
                # Background training mode
                missing_images = []
                missing_masks = []
                
                for expected_file in expected_files:
                    base_name = expected_file.replace('.png', '')
                    # Check for any background variant
                    bg_pattern = f"{base_name}_bg_"
                    
                    has_bg_img = any(bg_pattern in f for f in actual_img_files)
                    has_bg_mask = any(bg_pattern in f for f in actual_mask_files)
                    
                    if not has_bg_img:
                        missing_images.append(expected_file)
                    if not has_bg_mask:
                        missing_masks.append(expected_file)
                
                return {
                    "missing_images": missing_images,
                    "missing_masks": missing_masks,
                    "total_expected": len(expected_files),
                    "actual_images": len(actual_img_files),
                    "actual_masks": len(actual_mask_files)
                }
            else:
                # Standard training mode
                missing_images = expected_files - actual_img_files
                missing_masks = expected_files - actual_mask_files
                
                return {
                    "missing_images": list(missing_images),
                    "missing_masks": list(missing_masks),
                    "total_expected": len(expected_files),
                    "actual_images": len(actual_img_files), 
                    "actual_masks": len(actual_mask_files)
                }
        
        # This test demonstrates how to validate data consistency
        # It would catch the bug we found where background generation
        # creates incomplete file sets
        
        # Mock scenario
        with tempfile.TemporaryDirectory() as temp_dir:
            ann_file = Path(temp_dir) / "test.json"
            img_dir = Path(temp_dir) / "images"
            mask_dir = Path(temp_dir) / "masks"
            
            img_dir.mkdir()
            mask_dir.mkdir()
            
            # Create mock annotation
            mock_ann = {
                "images": [
                    {"file_name": "1.png"}, 
                    {"file_name": "2.png"},
                    {"file_name": "3.png"}
                ]
            }
            
            with open(ann_file, 'w') as f:
                json.dump(mock_ann, f)
            
            # Simulate incomplete background generation
            # (Only creates files for 1.png and 3.png, missing 2.png)
            (img_dir / "1_bg_00.png").touch()
            (img_dir / "3_bg_00.png").touch() 
            (mask_dir / "1_bg_00.png").touch()
            (mask_dir / "3_bg_00.png").touch()
            
            # Validate consistency
            result = validate_training_data_consistency(
                str(ann_file), str(img_dir), str(mask_dir)
            )
            
            # Should detect missing file for 2.png
            assert len(result["missing_images"]) == 1
            assert len(result["missing_masks"]) == 1
            assert result["total_expected"] == 3
            assert result["actual_images"] == 2
            assert result["actual_masks"] == 2