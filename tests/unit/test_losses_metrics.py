import pytest
import torch
import torch.nn.functional as F
import numpy as np
from unittest.mock import patch


class TestInpaintingLoss:
    """Test cases for inpainting loss function"""
    
    @pytest.fixture
    def loss_inputs(self):
        """Create sample inputs for loss testing"""
        batch_size = 2
        height, width = 64, 64
        
        # Create realistic test data
        generated = torch.randn(batch_size, 1, height, width) * 0.5
        real = torch.randn(batch_size, 1, height, width) * 0.5
        mask = torch.zeros(batch_size, 1, height, width)
        mask[:, :, 16:48, 16:48] = 1.0  # Square mask in center
        
        return generated, real, mask
    
    def test_l1_loss_calculation(self, loss_inputs):
        """Test L1 loss component calculation"""
        gen, real, mask = loss_inputs
        
        # Calculate masked and unmasked L1 losses
        masked_l1 = F.l1_loss(gen * mask, real * mask)
        unmasked_l1 = F.l1_loss(gen * (1 - mask), real * (1 - mask))
        
        # Combined weighted L1 loss (6x masked + 1x unmasked)
        total_l1 = masked_l1 * 6.0 + unmasked_l1 * 1.0
        
        assert isinstance(total_l1, torch.Tensor)
        assert total_l1.dim() == 0  # Scalar
        assert total_l1 >= 0.0
        assert not torch.isnan(total_l1)
        assert not torch.isinf(total_l1)
    
    def test_ssim_loss_calculation(self, loss_inputs):
        """Test SSIM loss component calculation"""
        from skimage.metrics import structural_similarity as ssim
        
        gen, real, mask = loss_inputs
        
        # Convert to numpy for SSIM calculation
        gen_np = gen[0, 0].detach().numpy()  # First sample, first channel
        real_np = real[0, 0].detach().numpy()
        
        # Calculate SSIM
        ssim_value = ssim(gen_np, real_np, data_range=2.0)  # data_range for [-1,1]
        ssim_loss = 1 - ssim_value
        
        assert 0.0 <= ssim_value <= 1.0
        assert 0.0 <= ssim_loss <= 1.0
        assert isinstance(ssim_loss, (float, np.float64))
    
    def test_combined_inpainting_loss(self, loss_inputs):
        """Test complete inpainting loss calculation"""
        from skimage.metrics import structural_similarity as ssim
        
        gen, real, mask = loss_inputs
        
        # L1 loss component
        l1_masked = F.l1_loss(gen * mask, real * mask) * 6.0
        l1_unmasked = F.l1_loss(gen * (1 - mask), real * (1 - mask)) * 1.0
        l1_loss = l1_masked + l1_unmasked
        
        # SSIM loss component (for first sample)
        gen_np = gen[0, 0].detach().numpy()
        real_np = real[0, 0].detach().numpy()
        ssim_value = ssim(gen_np, real_np, data_range=2.0, channel_axis=None)
        ssim_loss = (1 - ssim_value) * 0.5
        
        # Combined loss
        total_loss = l1_loss + ssim_loss
        
        assert total_loss >= 0.0
        assert not torch.isnan(torch.tensor(total_loss))
        assert not torch.isinf(torch.tensor(total_loss))
    
    def test_loss_with_perfect_reconstruction(self):
        """Test loss with identical generated and real images"""
        # Create identical tensors
        real = torch.randn(1, 1, 64, 64)
        gen = real.clone()
        mask = torch.zeros(1, 1, 64, 64)
        mask[:, :, 20:40, 20:40] = 1.0
        
        # L1 loss should be zero
        l1_loss = F.l1_loss(gen * mask, real * mask) * 6.0 + \
                 F.l1_loss(gen * (1 - mask), real * (1 - mask)) * 1.0
        
        assert torch.allclose(l1_loss, torch.tensor(0.0), atol=1e-6)
    
    def test_loss_gradient_flow(self, loss_inputs):
        """Test that loss function allows gradient flow"""
        gen, real, mask = loss_inputs
        gen.requires_grad_(True)
        
        # Calculate L1 loss only for gradient test
        l1_loss = F.l1_loss(gen * mask, real * mask) * 6.0 + \
                 F.l1_loss(gen * (1 - mask), real * (1 - mask)) * 1.0
        
        l1_loss.backward()
        
        assert gen.grad is not None
        assert not torch.any(torch.isnan(gen.grad))
        assert not torch.any(torch.isinf(gen.grad))
    
    def test_loss_with_different_mask_ratios(self):
        """Test loss behavior with different mask coverage ratios"""
        img_size = 64
        real = torch.randn(1, 1, img_size, img_size)
        gen = torch.randn(1, 1, img_size, img_size)
        
        mask_ratios = [0.1, 0.25, 0.5, 0.75, 0.9]
        losses = []
        
        for ratio in mask_ratios:
            mask = torch.zeros(1, 1, img_size, img_size)
            mask_size = int(img_size * np.sqrt(ratio))
            start = (img_size - mask_size) // 2
            mask[:, :, start:start+mask_size, start:start+mask_size] = 1.0
            
            l1_loss = F.l1_loss(gen * mask, real * mask) * 6.0 + \
                     F.l1_loss(gen * (1 - mask), real * (1 - mask)) * 1.0
            losses.append(l1_loss.item())
        
        # All losses should be valid
        for loss in losses:
            assert loss >= 0.0
            assert not np.isnan(loss)
            assert not np.isinf(loss)


class TestMetrics:
    """Test cases for evaluation metrics (PSNR, SSIM, Wasserstein, RMSE)"""
    
    def test_psnr_calculation(self):
        """Test PSNR calculation"""
        from src.utils import calculate_psnr
        
        # Perfect reconstruction
        img1 = torch.randn(64, 64)
        img2 = img1.clone()
        psnr = calculate_psnr(img1, img2)
        assert psnr >= 100  # Should be very high for identical images
        
        # Different images
        img3 = torch.randn(64, 64)
        psnr = calculate_psnr(img1, img3)
        assert 0 <= psnr <= 100  # Typical PSNR range
        assert not np.isnan(psnr)
        assert not np.isinf(psnr)
    
    def test_ssim_calculation(self):
        """Test SSIM calculation"""
        from src.utils import calculate_ssim
        
        # Perfect reconstruction
        img1 = torch.randn(64, 64)
        img2 = img1.clone()
        ssim = calculate_ssim(img1, img2)
        assert np.isclose(ssim, 1.0, atol=1e-6)
        
        # Different images
        img3 = torch.randn(64, 64)
        ssim = calculate_ssim(img1, img3)
        assert -1.0 <= ssim <= 1.0
        assert not np.isnan(ssim)
        assert not np.isinf(ssim)
    
    def test_wasserstein_distance(self):
        """Test Wasserstein distance calculation"""
        from src.utils import calculate_wasserstein
        
        # Identical distributions
        img1 = torch.randn(64, 64)
        img2 = img1.clone()
        wd = calculate_wasserstein(img1, img2)
        assert np.isclose(wd, 0.0, atol=1e-6)
        
        # Different distributions
        img3 = torch.randn(64, 64) + 1.0  # Shifted distribution
        wd = calculate_wasserstein(img1, img3)
        assert wd >= 0.0
        assert not np.isnan(wd)
        assert not np.isinf(wd)
    
    def test_rmse_calculation(self):
        """Test RMSE calculation"""
        from src.utils import calculate_rmse
        
        # Perfect reconstruction
        img1 = torch.randn(64, 64)
        img2 = img1.clone()
        rmse = calculate_rmse(img1, img2)
        assert np.isclose(rmse, 0.0, atol=1e-6)
        
        # Different images
        img3 = torch.randn(64, 64)
        rmse = calculate_rmse(img1, img3)
        assert rmse >= 0.0
        assert not np.isnan(rmse)
        assert not np.isinf(rmse)
    
    @pytest.mark.parametrize("metric_name", ["psnr", "ssim", "wasserstein", "rmse"])
    def test_metrics_with_batch_inputs(self, metric_name):
        """Test metrics with batch inputs"""
        from src.utils import calculate_psnr, calculate_ssim, calculate_wasserstein, calculate_rmse
        
        metric_funcs = {
            "psnr": calculate_psnr,
            "ssim": calculate_ssim, 
            "wasserstein": calculate_wasserstein,
            "rmse": calculate_rmse
        }
        
        # Create batch of images
        batch_size = 4
        img1_batch = torch.randn(batch_size, 64, 64)
        img2_batch = torch.randn(batch_size, 64, 64)
        
        metric_func = metric_funcs[metric_name]
        
        # Calculate metric for each sample in batch
        results = []
        for i in range(batch_size):
            result = metric_func(img1_batch[i], img2_batch[i])
            results.append(result)
            assert not np.isnan(result)
            assert not np.isinf(result)
        
        # Results should be consistent
        assert len(results) == batch_size
    
    def test_metrics_with_different_ranges(self):
        """Test metrics with different value ranges"""
        from src.utils import calculate_psnr, calculate_ssim
        
        # Test with [-1, 1] range (model output)
        img1 = torch.randn(64, 64).clamp(-1, 1)
        img2 = torch.randn(64, 64).clamp(-1, 1)
        
        psnr = calculate_psnr(img1, img2)
        ssim = calculate_ssim(img1, img2)
        
        assert 0 <= psnr <= 100
        assert -1 <= ssim <= 1
        
        # Test with [0, 255] range (typical image)
        img1_255 = (img1 + 1) * 127.5  # Convert to [0, 255]
        img2_255 = (img2 + 1) * 127.5
        
        psnr_255 = calculate_psnr(img1_255, img2_255)
        assert 0 <= psnr_255 <= 100


class TestLossComponents:
    """Test individual components of the loss function"""
    
    def test_masked_region_penalty(self):
        """Test that masked regions have higher penalty in loss"""
        real = torch.zeros(1, 1, 64, 64)
        gen_good = torch.zeros(1, 1, 64, 64)  # Perfect reconstruction
        gen_bad = torch.ones(1, 1, 64, 64)    # Poor reconstruction
        
        mask = torch.zeros(1, 1, 64, 64)
        mask[:, :, 20:40, 20:40] = 1.0
        
        # Loss for good reconstruction
        loss_good = F.l1_loss(gen_good * mask, real * mask) * 6.0 + \
                   F.l1_loss(gen_good * (1 - mask), real * (1 - mask)) * 1.0
        
        # Loss for bad reconstruction
        loss_bad = F.l1_loss(gen_bad * mask, real * mask) * 6.0 + \
                  F.l1_loss(gen_bad * (1 - mask), real * (1 - mask)) * 1.0
        
        assert loss_bad > loss_good
        assert torch.isclose(loss_good, torch.tensor(0.0), atol=1e-6)
    
    def test_unmasked_region_preservation(self):
        """Test that same magnitude error in masked vs unmasked regions is penalized differently"""
        real = torch.zeros(1, 1, 64, 64)
        mask = torch.zeros(1, 1, 64, 64)
        mask[:, :, 20:40, 20:40] = 1.0
        
        # Same magnitude error (1.0) in masked region only
        gen_masked_error = torch.zeros(1, 1, 64, 64)
        gen_masked_error[:, :, 20:40, 20:40] = 1.0
        
        # Same magnitude error (1.0) in a similar sized unmasked region
        gen_unmasked_error = torch.zeros(1, 1, 64, 64)
        gen_unmasked_error[:, :, 0:20, 0:20] = 1.0  # Same size region but unmasked
        
        # Calculate loss for error in masked region
        loss_masked = F.l1_loss(gen_masked_error * mask, real * mask) * 6.0 + \
                     F.l1_loss(gen_masked_error * (1 - mask), real * (1 - mask)) * 1.0
        
        # Calculate loss for error in unmasked region  
        loss_unmasked = F.l1_loss(gen_unmasked_error * mask, real * mask) * 6.0 + \
                       F.l1_loss(gen_unmasked_error * (1 - mask), real * (1 - mask)) * 1.0
        
        # Error in masked region should be penalized more heavily (6x vs 1x)
        assert loss_masked > loss_unmasked
    
    def test_ssim_data_range(self):
        """Test SSIM with correct data range for normalized images"""
        from skimage.metrics import structural_similarity as ssim
        
        # Images in [-1, 1] range
        img1 = torch.randn(64, 64).clamp(-1, 1)
        img2 = img1 + torch.randn(64, 64) * 0.1  # Small perturbation
        
        # Convert to numpy
        img1_np = img1.numpy()
        img2_np = img2.numpy()
        
        # SSIM with correct data range
        ssim_value = ssim(img1_np, img2_np, data_range=2.0)  # Range is 2.0 for [-1,1]
        
        assert 0.0 <= ssim_value <= 1.0
        assert ssim_value > 0.5  # Should be high due to small perturbation
    
    def test_loss_numerical_stability(self):
        """Test loss function numerical stability with extreme values"""
        # Very small values
        small_real = torch.ones(1, 1, 64, 64) * 1e-8
        small_gen = torch.ones(1, 1, 64, 64) * 1e-8
        mask = torch.zeros(1, 1, 64, 64)
        mask[:, :, 20:40, 20:40] = 1.0
        
        loss = F.l1_loss(small_gen * mask, small_real * mask) * 6.0 + \
               F.l1_loss(small_gen * (1 - mask), small_real * (1 - mask)) * 1.0
        
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        assert loss >= 0.0
        
        # Large values (within model output range)
        large_real = torch.ones(1, 1, 64, 64) * 0.99
        large_gen = torch.ones(1, 1, 64, 64) * (-0.99)
        
        loss_large = F.l1_loss(large_gen * mask, large_real * mask) * 6.0 + \
                    F.l1_loss(large_gen * (1 - mask), large_real * (1 - mask)) * 1.0
        
        assert not torch.isnan(loss_large)
        assert not torch.isinf(loss_large)
        assert loss_large >= 0.0