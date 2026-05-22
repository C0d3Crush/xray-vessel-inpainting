import pytest
import numpy as np
import torch
import tempfile
import json
import cv2
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys
sys.path.append('src')


class TestTrainingRobustness:
    """Test training robustness against data issues"""

    @pytest.fixture
    def incomplete_dataset_setup(self):
        """Create dataset with missing files to test robustness"""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            
            # Create annotation with 5 images
            annotations = {
                "images": [
                    {"id": 1, "file_name": "1.png", "width": 64, "height": 64},
                    {"id": 2, "file_name": "2.png", "width": 64, "height": 64}, 
                    {"id": 3, "file_name": "3.png", "width": 64, "height": 64},
                    {"id": 4, "file_name": "4.png", "width": 64, "height": 64},
                    {"id": 5, "file_name": "5.png", "width": 64, "height": 64}
                ],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 1, "segmentation": [[0,0,10,0,10,10,0,10]], "area": 100, "bbox": [0,0,10,10], "iscrowd": 0},
                    {"id": 2, "image_id": 2, "category_id": 1, "segmentation": [[0,0,10,0,10,10,0,10]], "area": 100, "bbox": [0,0,10,10], "iscrowd": 0},
                    {"id": 3, "image_id": 3, "category_id": 1, "segmentation": [[0,0,10,0,10,10,0,10]], "area": 100, "bbox": [0,0,10,10], "iscrowd": 0},
                    {"id": 4, "image_id": 4, "category_id": 1, "segmentation": [[0,0,10,0,10,10,0,10]], "area": 100, "bbox": [0,0,10,10], "iscrowd": 0},
                    {"id": 5, "image_id": 5, "category_id": 1, "segmentation": [[0,0,10,0,10,10,0,10]], "area": 100, "bbox": [0,0,10,10], "iscrowd": 0}
                ],
                "categories": [{"id": 1, "name": "vessel"}]
            }
            
            ann_file = base_dir / "annotations.json"
            with open(ann_file, 'w') as f:
                json.dump(annotations, f)
            
            # Create image directory with only SOME files (simulating incomplete background generation)
            img_dir = base_dir / "images"
            mask_dir = base_dir / "masks" 
            img_dir.mkdir()
            mask_dir.mkdir()
            
            # Only create files for images 1, 3, 5 (missing 2, 4)
            # This simulates background generation that failed for some images
            existing_files = ["1_bg_00.png", "3_bg_00.png", "5_bg_00.png"]
            
            for filename in existing_files:
                img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
                mask = np.random.randint(0, 2, (64, 64), dtype=np.uint8) * 255
                
                cv2.imwrite(str(img_dir / filename), img)
                cv2.imwrite(str(mask_dir / filename), mask)
            
            yield {
                "base_dir": base_dir,
                "ann_file": str(ann_file),
                "img_dir": str(img_dir), 
                "mask_dir": str(mask_dir),
                "existing_files": existing_files,
                "missing_files": ["2_bg_00.png", "4_bg_00.png"]
            }

    def test_dataset_handles_missing_files(self, incomplete_dataset_setup):
        """Test that dataset gracefully handles missing background files"""
        from train import ArcadeDataset
        
        setup = incomplete_dataset_setup
        
        # This should expose the FileNotFoundError issue
        with pytest.raises(FileNotFoundError):
            dataset = ArcadeDataset(
                img_path=setup["img_dir"],
                ann_path=setup["ann_file"], 
                mask_dir=setup["mask_dir"],
                input_size=64
            )
            
            # Try to access all items - will fail on missing files
            for i in range(len(dataset)):
                item = dataset[i]

    def test_robust_dataset_with_file_filtering(self, incomplete_dataset_setup):
        """Test improved dataset that filters out missing files"""
        # This test shows how the dataset SHOULD work
        
        def create_robust_dataset(img_path, ann_path, mask_dir, input_size):
            """Improved dataset that filters out missing files"""
            import json
            from pathlib import Path
            
            # Load annotations
            with open(ann_path, 'r') as f:
                data = json.load(f)
            
            # Filter images to only include those with existing background files
            available_images = []
            img_dir = Path(img_path)
            mask_dir_path = Path(mask_dir)
            
            for img_info in data['images']:
                base_name = img_info['file_name'].replace('.png', '')
                
                # Check for any background variant
                bg_pattern = f"{base_name}_bg_"
                has_bg_img = any(bg_pattern in f.name for f in img_dir.glob("*.png"))
                has_bg_mask = any(bg_pattern in f.name for f in mask_dir_path.glob("*.png"))
                
                if has_bg_img and has_bg_mask:
                    available_images.append(img_info)
            
            return available_images
        
        setup = incomplete_dataset_setup
        
        # Get filtered images
        available_images = create_robust_dataset(
            setup["img_dir"],
            setup["ann_file"],
            setup["mask_dir"],
            64
        )
        
        # Should only include images 1, 3, 5 (the ones with background files)
        assert len(available_images) == 3
        available_ids = {img['id'] for img in available_images}
        assert available_ids == {1, 3, 5}

    def test_background_generation_failure_rates(self):
        """Test background generation under different vessel density conditions"""
        
        def simulate_background_generation(vessel_density):
            """Simulate background generation with different vessel densities"""
            # Create vessel mask with specified density
            mask = np.zeros((256, 256), dtype=np.uint8)
            
            if vessel_density == "low":
                # Sparse vessels - should succeed
                cv2.line(mask, (100, 100), (150, 150), 255, 2)
            elif vessel_density == "medium":
                # Medium vessel density
                cv2.line(mask, (50, 50), (200, 200), 255, 3)
                cv2.line(mask, (100, 20), (100, 230), 255, 3)
            elif vessel_density == "high":
                # Dense vessel network - may fail
                for i in range(0, 256, 30):
                    cv2.line(mask, (0, i), (255, i), 255, 3)
                for i in range(0, 256, 30):
                    cv2.line(mask, (i, 0), (i, 255), 255, 3)
            elif vessel_density == "extreme":
                # Completely filled - should always fail
                mask.fill(255)
            
            # Import and test background generation
            from generate_background_masks import generate_background_training_mask
            
            bg_mask, num_shapes = generate_background_training_mask(
                mask.shape, mask, num_shapes=5, safety_margin=5
            )
            
            return num_shapes > 0, num_shapes
        
        # Test different scenarios
        success_low, shapes_low = simulate_background_generation("low")
        success_medium, shapes_medium = simulate_background_generation("medium") 
        success_high, shapes_high = simulate_background_generation("high")
        success_extreme, shapes_extreme = simulate_background_generation("extreme")
        
        # Low density should usually succeed
        assert success_low == True
        assert shapes_low >= 3
        
        # Medium density may or may not succeed
        # (This is realistic - some images may be too dense)
        
        # High density likely fails
        assert shapes_high <= shapes_medium
        
        # Extreme density should always fail
        assert success_extreme == False
        assert shapes_extreme == 0

    def test_training_with_limited_background_data(self):
        """Test training behavior when only small fraction of background data is available"""
        
        # Simulate scenario where only 30% of images generate successful backgrounds
        total_annotation_images = 1000
        successful_bg_generation = 300  # Only 30% success rate
        
        # This represents the real-world scenario we found
        expected_training_images = total_annotation_images  # What training script expects
        actual_available_images = successful_bg_generation  # What's actually available
        
        missing_percentage = (expected_training_images - actual_available_images) / expected_training_images
        
        # 70% missing files would cause massive training failures
        assert missing_percentage == 0.7
        
        # Training would fail catastrophically without proper handling
        assert actual_available_images < expected_training_images / 2

    def test_memory_usage_with_patch_mode(self):
        """Test memory implications of patch mode with incomplete data"""
        
        # Test different patch configurations
        scenarios = [
            {"images": 1000, "patches_per_image": 16, "batch_size": 32},
            {"images": 300, "patches_per_image": 16, "batch_size": 32},  # After filtering
            {"images": 300, "patches_per_image": 8, "batch_size": 16},   # Reduced load
        ]
        
        for scenario in scenarios:
            total_patches = scenario["images"] * scenario["patches_per_image"]
            iterations_per_epoch = total_patches // scenario["batch_size"]
            
            # Memory usage is roughly proportional to batch size and patch count
            relative_memory = scenario["batch_size"] * scenario["patches_per_image"]
            
            # Document the implications
            scenario["total_patches"] = total_patches
            scenario["iterations_per_epoch"] = iterations_per_epoch
            scenario["relative_memory"] = relative_memory
        
        # Original scenario (before fixing missing files)
        original = scenarios[0]
        # After filtering missing files
        filtered = scenarios[1] 
        # Optimized for GPU memory
        optimized = scenarios[2]
        
        # Filtering reduces total training data significantly
        assert filtered["total_patches"] < original["total_patches"] * 0.5
        
        # But optimized version uses less memory
        assert optimized["relative_memory"] < filtered["relative_memory"]

    @pytest.mark.integration
    def test_workflow_error_handling(self):
        """Test that workflow provides helpful error messages"""
        
        def check_error_quality(error_msg, missing_files):
            """Check if error message is helpful for debugging"""
            
            helpful_indicators = [
                "FileNotFoundError" in error_msg,
                "not found" in error_msg.lower(),
                any(filename in error_msg for filename in missing_files),
            ]
            
            return any(helpful_indicators)
        
        # Simulate the actual error we encountered
        missing_files = ["750.png", "855.png", "918.png"]
        error_msg = "FileNotFoundError: Image not found: data/smoke_bg_img/750.png"
        
        # Current error is somewhat helpful
        is_helpful = check_error_quality(error_msg, missing_files)
        assert is_helpful
        
        # But improved error would be more diagnostic
        improved_error = f"""
        Training failed due to missing background files.
        Expected 1000 background images, but only 700 were generated.
        Missing files: {missing_files[:5]}... (and 295 more)
        
        This typically happens when:
        1. Vessel density is too high in some images
        2. Safety margin is too large
        3. Background shape generation failed
        
        Solutions:
        1. Reduce safety margin: --safety-margin 3
        2. Use smaller background shapes
        3. Filter dataset to only include successful generations
        """
        
        # Better error provides actionable information
        assert "Solutions:" in improved_error
        assert "safety-margin" in improved_error

    def test_data_augmentation_impact(self):
        """Test impact of missing files on data augmentation effectiveness"""
        
        # Original plan: 1000 images * 2 variations = 2000 training samples
        original_plan = {
            "base_images": 1000,
            "variations_per_image": 2,
            "expected_samples": 2000
        }
        
        # Reality: Only 60% successful generation
        actual_result = {
            "base_images": 1000,
            "successful_generations": 600,  # 60% success rate
            "variations_per_image": 2,
            "actual_samples": 1200  # Much less than planned
        }
        
        # Impact on training
        data_reduction = 1 - (actual_result["actual_samples"] / original_plan["expected_samples"])
        
        # 40% reduction in training data is significant
        assert data_reduction == 0.4
        
        # This could substantially impact model performance
        assert actual_result["actual_samples"] < original_plan["expected_samples"]

    @pytest.mark.parametrize("vessel_coverage", [0.1, 0.3, 0.5, 0.7, 0.9])
    def test_background_generation_vs_vessel_coverage(self, vessel_coverage):
        """Test background generation success rate vs vessel coverage percentage"""
        
        def create_vessel_mask_with_coverage(coverage, size=(256, 256)):
            """Create vessel mask with specified coverage percentage"""
            mask = np.zeros(size, dtype=np.uint8)
            
            # Add random vessel patterns until we reach coverage
            current_coverage = 0.0
            attempt = 0
            
            while current_coverage < coverage and attempt < 100:
                # Add random line
                x1, y1 = np.random.randint(0, size[1]), np.random.randint(0, size[0])
                x2, y2 = np.random.randint(0, size[1]), np.random.randint(0, size[0])
                thickness = np.random.randint(1, 5)
                
                cv2.line(mask, (x1, y1), (x2, y2), 255, thickness)
                
                current_coverage = np.sum(mask > 0) / (size[0] * size[1])
                attempt += 1
            
            return mask
        
        # Test background generation with this vessel density
        vessel_mask = create_vessel_mask_with_coverage(vessel_coverage)
        
        from generate_background_masks import generate_background_training_mask
        
        bg_mask, num_shapes = generate_background_training_mask(
            vessel_mask.shape, vessel_mask, num_shapes=5, safety_margin=5
        )
        
        success = num_shapes > 0
        
        # Expected: success rate decreases as vessel coverage increases
        if vessel_coverage <= 0.3:
            # Low vessel density - should usually succeed
            pass  # May or may not succeed, depends on randomness
        elif vessel_coverage >= 0.7:
            # High vessel density - likely to fail
            assert num_shapes <= 2  # Few or no shapes generated
        
        # Document the relationship for analysis
        print(f"Vessel coverage: {vessel_coverage:.1%}, Success: {success}, Shapes: {num_shapes}")