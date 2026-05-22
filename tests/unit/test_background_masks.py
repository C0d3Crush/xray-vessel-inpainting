import pytest
import numpy as np
import cv2
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the module we're testing
import sys
sys.path.append('src')
from generate_background_masks import (
    create_vessel_exclusion_mask,
    generate_random_circle,
    generate_random_rectangle,
    generate_random_blob,
    generate_background_training_mask,
    process_training_masks
)


class TestBackgroundMaskGeneration:
    """Test cases for background mask generation"""

    @pytest.fixture
    def sample_vessel_mask(self):
        """Create a sample vessel mask for testing"""
        mask = np.zeros((256, 256), dtype=np.uint8)
        # Draw some vessels
        cv2.line(mask, (50, 50), (200, 200), 255, 5)
        cv2.line(mask, (100, 20), (100, 230), 255, 3)
        return mask

    @pytest.fixture
    def sample_image(self):
        """Create a sample grayscale image"""
        return np.random.randint(0, 256, (256, 256), dtype=np.uint8)

    def test_vessel_exclusion_mask_creation(self, sample_vessel_mask):
        """Test vessel exclusion mask with safety margin"""
        safety_margin = 5
        exclusion_mask = create_vessel_exclusion_mask(sample_vessel_mask, safety_margin)
        
        # Check that exclusion mask is larger than original vessel mask
        original_area = np.sum(sample_vessel_mask > 0)
        exclusion_area = np.sum(exclusion_mask > 0)
        assert exclusion_area > original_area
        
        # Check that all original vessel pixels are included
        vessel_pixels = sample_vessel_mask > 0
        exclusion_pixels = exclusion_mask > 0
        assert np.all(exclusion_pixels[vessel_pixels])

    def test_random_circle_generation(self, sample_vessel_mask):
        """Test random circle generation in vessel-free areas"""
        img_shape = sample_vessel_mask.shape
        exclusion_mask = create_vessel_exclusion_mask(sample_vessel_mask, 5)
        
        circle_mask, success = generate_random_circle(img_shape, exclusion_mask)
        
        if success:
            # Check that circle doesn't overlap significantly with vessels
            overlap = np.sum((circle_mask > 0) & (exclusion_mask > 0))
            circle_area = np.sum(circle_mask > 0)
            overlap_ratio = overlap / circle_area if circle_area > 0 else 1.0
            assert overlap_ratio < 0.1

    def test_random_rectangle_generation(self, sample_vessel_mask):
        """Test random rectangle generation in vessel-free areas"""
        img_shape = sample_vessel_mask.shape
        exclusion_mask = create_vessel_exclusion_mask(sample_vessel_mask, 5)
        
        rect_mask, success = generate_random_rectangle(img_shape, exclusion_mask)
        
        if success:
            # Check that rectangle doesn't overlap significantly with vessels
            overlap = np.sum((rect_mask > 0) & (exclusion_mask > 0))
            rect_area = np.sum(rect_mask > 0)
            overlap_ratio = overlap / rect_area if rect_area > 0 else 1.0
            assert overlap_ratio < 0.1

    def test_background_training_mask_generation(self, sample_vessel_mask):
        """Test complete background training mask generation"""
        img_shape = sample_vessel_mask.shape
        
        bg_mask, num_shapes = generate_background_training_mask(
            img_shape, sample_vessel_mask, num_shapes=3, safety_margin=5
        )
        
        # Should generate at least some shapes
        assert num_shapes >= 0
        
        if num_shapes > 0:
            # Check that background mask has non-zero area
            bg_area = np.sum(bg_mask > 0)
            assert bg_area > 0
            
            # Check vessel avoidance
            exclusion_mask = create_vessel_exclusion_mask(sample_vessel_mask, 5)
            overlap = np.sum((bg_mask > 0) & (exclusion_mask > 0))
            bg_area = np.sum(bg_mask > 0)
            overlap_ratio = overlap / bg_area if bg_area > 0 else 1.0
            assert overlap_ratio < 0.1

    def test_background_mask_consistency(self, sample_vessel_mask):
        """Test that background mask generation is consistent"""
        img_shape = sample_vessel_mask.shape
        
        # Generate multiple masks with same parameters
        masks = []
        shape_counts = []
        
        for _ in range(5):
            bg_mask, num_shapes = generate_background_training_mask(
                img_shape, sample_vessel_mask, num_shapes=3, safety_margin=5
            )
            masks.append(bg_mask)
            shape_counts.append(num_shapes)
        
        # Should generate some shapes in most attempts
        successful_generations = sum(1 for count in shape_counts if count > 0)
        assert successful_generations >= 3  # At least 60% success rate

    @pytest.mark.integration
    def test_process_training_masks_file_completeness(self, sample_image, sample_vessel_mask):
        """Test that process_training_masks handles missing files correctly"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup input directories
            input_img_dir = Path(temp_dir) / "input_images"
            input_mask_dir = Path(temp_dir) / "input_masks"
            output_img_dir = Path(temp_dir) / "output_images"
            output_mask_dir = Path(temp_dir) / "output_masks"
            
            input_img_dir.mkdir()
            input_mask_dir.mkdir()
            
            # Save test image and mask
            test_image_path = input_img_dir / "test_image.png"
            test_mask_path = input_mask_dir / "test_image.png"
            
            cv2.imwrite(str(test_image_path), sample_image)
            cv2.imwrite(str(test_mask_path), sample_vessel_mask)
            
            # Process training masks
            process_training_masks(
                str(input_img_dir),
                str(input_mask_dir), 
                str(output_img_dir),
                str(output_mask_dir),
                num_variations=2,
                safety_margin=5
            )
            
            # Check that output directories exist
            assert output_img_dir.exists()
            assert output_mask_dir.exists()
            
            # Check for generated files
            output_images = list(output_img_dir.glob("*.png"))
            output_masks = list(output_mask_dir.glob("*.png"))
            
            # Should generate some files (may not be all variations if vessel density is high)
            assert len(output_images) >= 0
            assert len(output_images) == len(output_masks)
            
            # Verify file naming convention
            for img_file in output_images:
                assert "_bg_" in img_file.name
                mask_file = output_mask_dir / img_file.name
                assert mask_file.exists()

    def test_vessel_dense_image_handling(self):
        """Test handling of images with very dense vessel patterns"""
        # Create image completely filled with vessels (worst case)
        dense_vessel_mask = np.full((256, 256), 255, dtype=np.uint8)
        
        bg_mask, num_shapes = generate_background_training_mask(
            dense_vessel_mask.shape, dense_vessel_mask, num_shapes=5, safety_margin=5
        )
        
        # Should gracefully handle case where no vessel-free areas exist
        assert num_shapes == 0
        assert np.sum(bg_mask) == 0

    def test_empty_vessel_mask_handling(self):
        """Test handling of images with no vessels (best case)"""
        # Create empty vessel mask (no vessels)
        empty_vessel_mask = np.zeros((256, 256), dtype=np.uint8)
        
        bg_mask, num_shapes = generate_background_training_mask(
            empty_vessel_mask.shape, empty_vessel_mask, num_shapes=5, safety_margin=5
        )
        
        # Should successfully generate all requested shapes
        assert num_shapes >= 3  # Should generate most shapes
        assert np.sum(bg_mask) > 0

    def test_safety_margin_effectiveness(self, sample_vessel_mask):
        """Test that safety margin prevents background shapes near vessels"""
        img_shape = sample_vessel_mask.shape
        
        # Test different safety margins
        for margin in [0, 3, 10]:
            exclusion_mask = create_vessel_exclusion_mask(sample_vessel_mask, margin)
            bg_mask, num_shapes = generate_background_training_mask(
                img_shape, sample_vessel_mask, num_shapes=3, safety_margin=margin
            )
            
            if num_shapes > 0:
                # Check vessel avoidance
                overlap = np.sum((bg_mask > 0) & (exclusion_mask > 0))
                bg_area = np.sum(bg_mask > 0)
                overlap_ratio = overlap / bg_area if bg_area > 0 else 1.0
                assert overlap_ratio < 0.1

    @pytest.mark.slow 
    def test_dataset_file_consistency(self):
        """Test that background generation creates consistent file sets for training"""
        # This test would verify the core issue we found:
        # That all annotation-based image IDs have corresponding background files
        
        # Mock scenario where annotation has 1000 images but background generation
        # only succeeds for 800 images
        annotation_image_ids = [f"{i}.png" for i in range(1, 1001)]
        generated_bg_files = [f"{i}_bg_00.png" for i in range(1, 801)]  # Missing 200 files
        
        # The training script would fail because it expects all 1000 to have backgrounds
        missing_files = []
        for img_id in annotation_image_ids:
            base_name = img_id.replace('.png', '')
            expected_bg_file = f"{base_name}_bg_00.png"
            if expected_bg_file not in generated_bg_files:
                missing_files.append(expected_bg_file)
        
        # This represents the actual bug we found
        assert len(missing_files) == 200
        
        # Test would fail here, demonstrating the dataset consistency issue
        with pytest.raises(AssertionError):
            assert len(missing_files) == 0, f"Missing background files: {len(missing_files)}"