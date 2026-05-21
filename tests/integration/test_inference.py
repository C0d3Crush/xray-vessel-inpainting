import pytest
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import json
import tempfile
import shutil


@pytest.mark.integration
class TestInferencePipeline:
    """Integration tests for the inference pipeline"""
    
    @pytest.fixture
    def inference_setup(self, temp_dir, sample_checkpoint):
        """Set up inference environment with model and test data"""
        # Create test images
        img_dir = temp_dir / "test_images"
        mask_dir = temp_dir / "test_masks"
        output_dir = temp_dir / "outputs"
        
        for dir_path in [img_dir, mask_dir, output_dir]:
            dir_path.mkdir(parents=True)
        
        # Create test image
        img_array = np.random.randint(50, 200, (256, 256), dtype=np.uint8)
        test_img = Image.fromarray(img_array, mode='L')
        test_img.save(img_dir / "test_001.png")
        
        # Create test mask
        mask_array = np.zeros((256, 256), dtype=np.uint8)
        mask_array[64:192, 64:192] = 255  # Square mask in center
        test_mask = Image.fromarray(mask_array, mode='L')
        test_mask.save(mask_dir / "test_001.png")
        
        return {
            "checkpoint": str(sample_checkpoint),
            "img_dir": str(img_dir),
            "mask_dir": str(mask_dir),
            "output_dir": str(output_dir),
            "test_img": str(img_dir / "test_001.png"),
            "test_mask": str(mask_dir / "test_001.png")
        }
    
    def test_demo_script_functionality(self, inference_setup):
        """Test core demo script functionality"""
        import sys
        sys.path.insert(0, str(Path.cwd() / "src"))
        
        from src.demo import load_model, process_image
        
        # Load model from checkpoint
        device = torch.device('cpu')
        model = load_model(inference_setup["checkpoint"], device=device, input_size=64)
        
        assert model is not None
        model.eval()
        
        # Load and process test image
        img_path = Path(inference_setup["test_img"])
        mask_path = Path(inference_setup["test_mask"])
        
        # Process single image
        result = process_image(model, img_path, mask_path, input_size=64, device=device)
        
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == (64, 64)  # Resized to input_size
        assert 0 <= result.min() and result.max() <= 255  # Valid image range
    
    def test_batch_inference(self, inference_setup):
        """Test inference on multiple images"""
        import sys
        sys.path.insert(0, str(Path.cwd() / "src"))
        
        from src.demo import load_model
        from src.network.network_pro import Inpaint
        
        # Create multiple test images
        img_dir = Path(inference_setup["img_dir"])
        mask_dir = Path(inference_setup["mask_dir"])
        
        for i in range(3):
            # Create varied test images
            img_array = np.random.randint(30 + i*20, 180 + i*20, (256, 256), dtype=np.uint8)
            test_img = Image.fromarray(img_array, mode='L')
            test_img.save(img_dir / f"test_{i:03d}.png")
            
            # Create varied masks
            mask_array = np.zeros((256, 256), dtype=np.uint8)
            start = 50 + i*10
            end = 200 - i*10
            mask_array[start:end, start:end] = 255
            test_mask = Image.fromarray(mask_array, mode='L')
            test_mask.save(mask_dir / f"test_{i:03d}.png")
        
        # Load model
        device = torch.device('cpu')
        model = load_model(inference_setup["checkpoint"], device=device, input_size=64)
        
        # Process all images
        img_files = sorted(img_dir.glob("*.png"))
        mask_files = sorted(mask_dir.glob("*.png"))
        
        assert len(img_files) == 4  # 3 new + 1 from fixture
        assert len(mask_files) == 4
        
        results = []
        for img_file, mask_file in zip(img_files, mask_files):
            from src.demo import process_image
            result = process_image(model, img_file, mask_file, input_size=64, device=device)
            results.append(result)
        
        # Verify all results
        assert len(results) == 4
        for result in results:
            assert result is not None
            assert isinstance(result, np.ndarray)
            assert result.shape == (64, 64)
            assert not np.any(np.isnan(result))
            assert not np.any(np.isinf(result))
    
    def test_inference_with_different_input_sizes(self, inference_setup):
        """Test inference with different input sizes"""
        import sys
        sys.path.insert(0, str(Path.cwd() / "src"))
        
        from src.demo import load_model, process_image
        
        device = torch.device('cpu')
        img_path = Path(inference_setup["test_img"])
        mask_path = Path(inference_setup["test_mask"])
        
        # Test different input sizes
        input_sizes = [32, 64, 128]
        
        for input_size in input_sizes:
            # Create model with specific input size
            from src.network.network_pro import Inpaint
            model = Inpaint(input_size=input_size, device=device)
            
            # Simulate loading checkpoint (just for testing)
            # In real scenario, would load actual checkpoint
            
            # Process image with this input size
            result = process_image(model, img_path, mask_path, input_size=input_size, device=device)
            
            assert result.shape == (input_size, input_size)
            assert 0 <= result.min() and result.max() <= 255
    
    def test_inference_preserves_unmasked_regions(self, inference_setup):
        """Test that inference preserves unmasked regions"""
        import sys
        sys.path.insert(0, str(Path.cwd() / "src"))
        
        from src.demo import load_model
        import torchvision.transforms as transforms
        
        device = torch.device('cpu')
        model = load_model(inference_setup["checkpoint"], device=device, input_size=64)
        
        # Load and preprocess image and mask
        img = Image.open(inference_setup["test_img"]).convert('L')
        mask = Image.open(inference_setup["test_mask"]).convert('L')
        
        # Resize to model input size
        resize_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # To [-1, 1]
        ])
        
        img_tensor = resize_transform(img).unsqueeze(0)  # Add batch dimension
        mask_tensor = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])(mask).unsqueeze(0)
        
        # Run inference
        model.eval()
        with torch.no_grad():
            output = model(img_tensor, mask_tensor)
        
        # Check preservation of unmasked regions
        unmasked_regions = (mask_tensor == 0).float()
        original_unmasked = img_tensor * unmasked_regions
        output_unmasked = output * unmasked_regions
        
        # Should be very similar in unmasked regions
        mse = torch.mean((original_unmasked - output_unmasked) ** 2)
        assert mse < 1e-4, f"Unmasked regions not preserved well, MSE: {mse}"
    
    def test_inference_output_quality(self, inference_setup):
        """Test basic quality metrics of inference output"""
        import sys
        sys.path.insert(0, str(Path.cwd() / "src"))
        
        from src.demo import load_model, process_image
        from src.utils import calculate_psnr, calculate_ssim
        
        device = torch.device('cpu')
        model = load_model(inference_setup["checkpoint"], device=device, input_size=64)
        
        img_path = Path(inference_setup["test_img"])
        mask_path = Path(inference_setup["test_mask"])
        
        # Get inference result
        result = process_image(model, img_path, mask_path, input_size=64, device=device)
        
        # Load original image for comparison
        original_img = Image.open(img_path).convert('L')
        original_resized = np.array(original_img.resize((64, 64)))
        
        # Calculate quality metrics
        psnr = calculate_psnr(
            torch.from_numpy(result).float(),
            torch.from_numpy(original_resized).float()
        )
        
        ssim = calculate_ssim(
            torch.from_numpy(result).float(),
            torch.from_numpy(original_resized).float()
        )
        
        # Basic quality checks
        assert psnr > 0, f"Invalid PSNR: {psnr}"
        assert -1 <= ssim <= 1, f"Invalid SSIM: {ssim}"
        assert not np.isnan(psnr)
        assert not np.isnan(ssim)
    
    def test_inference_with_edge_cases(self, temp_dir):
        """Test inference with edge case inputs"""
        import sys
        sys.path.insert(0, str(Path.cwd() / "src"))
        
        from src.network.network_pro import Inpaint
        
        device = torch.device('cpu')
        model = Inpaint(input_size=64, device=device)
        model.eval()
        
        test_cases = [
            # All black image
            (torch.zeros(1, 1, 64, 64), torch.zeros(1, 1, 64, 64)),
            # All white image  
            (torch.ones(1, 1, 64, 64), torch.zeros(1, 1, 64, 64)),
            # Full mask (everything masked)
            (torch.randn(1, 1, 64, 64) * 0.5, torch.ones(1, 1, 64, 64)),
            # No mask (nothing masked)
            (torch.randn(1, 1, 64, 64) * 0.5, torch.zeros(1, 1, 64, 64)),
            # Very small mask
            (torch.randn(1, 1, 64, 64) * 0.5, self._create_small_mask(64)),
        ]
        
        for img, mask in test_cases:
            with torch.no_grad():
                output = model(img, mask)
            
            # Check output validity
            assert output.shape == (1, 1, 64, 64)
            assert not torch.any(torch.isnan(output))
            assert not torch.any(torch.isinf(output))
            assert torch.all(output >= -1.1)  # Allow small numerical errors
            assert torch.all(output <= 1.1)
    
    def _create_small_mask(self, size):
        """Create a small mask for testing"""
        mask = torch.zeros(1, 1, size, size)
        center = size // 2
        mask[:, :, center-2:center+2, center-2:center+2] = 1.0
        return mask
    
    def test_inference_memory_usage(self, inference_setup):
        """Test memory usage during inference"""
        import sys
        sys.path.insert(0, str(Path.cwd() / "src"))
        
        from src.demo import load_model, process_image
        import gc
        
        device = torch.device('cpu')
        
        # Monitor memory before
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        # Load model and run inference
        model = load_model(inference_setup["checkpoint"], device=device, input_size=64)
        
        img_path = Path(inference_setup["test_img"])
        mask_path = Path(inference_setup["test_mask"])
        
        # Multiple inference runs to check for memory leaks
        for _ in range(5):
            result = process_image(model, img_path, mask_path, input_size=64, device=device)
            assert result is not None
        
        # Clean up
        del model
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Check memory usage (basic check for major leaks)
        final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        if torch.cuda.is_available():
            assert final_memory - initial_memory < 1024 * 1024 * 100  # Less than 100MB increase
    
    def test_inference_reproducibility(self, inference_setup):
        """Test that inference produces reproducible results"""
        import sys
        sys.path.insert(0, str(Path.cwd() / "src"))
        
        from src.demo import load_model, process_image
        
        device = torch.device('cpu')
        img_path = Path(inference_setup["test_img"])
        mask_path = Path(inference_setup["test_mask"])
        
        results = []
        
        # Run inference multiple times
        for i in range(3):
            # Set seeds for reproducibility
            torch.manual_seed(42)
            np.random.seed(42)
            
            model = load_model(inference_setup["checkpoint"], device=device, input_size=64)
            result = process_image(model, img_path, mask_path, input_size=64, device=device)
            results.append(result)
        
        # Results should be identical (or very close due to floating point)
        for i in range(1, len(results)):
            np.testing.assert_array_almost_equal(
                results[0], results[i], 
                decimal=5, 
                err_msg=f"Inference result {i} differs from first result"
            )