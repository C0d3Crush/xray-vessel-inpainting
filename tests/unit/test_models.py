import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock


class TestViTModel:
    """Test cases for ViT (Continuously Masked Transformer) model"""
    
    @pytest.fixture
    def vit_model(self, device):
        from src.network.vit import ViT
        return ViT(
            input_size=64,
            patch_size=4,
            embed_dim=768,
            depth=15,
            num_heads=16,
            mlp_ratio=4,
            device=device
        )
    
    def test_vit_initialization(self, vit_model, device):
        """Test ViT model initialization"""
        assert vit_model.device == device
        assert vit_model.input_size == 64
        assert vit_model.patch_size == 4
        assert vit_model.embed_dim == 768
        assert vit_model.depth == 15
        
    def test_vit_forward_pass(self, vit_model, sample_image, sample_mask):
        """Test ViT forward pass with masked input"""
        # Resize to match model input size
        img = torch.nn.functional.interpolate(sample_image, size=(64, 64))
        mask = torch.nn.functional.interpolate(sample_mask, size=(64, 64))
        
        # Create masked input (img * (1 - mask))
        masked_img = img * (1 - mask)
        
        with torch.no_grad():
            output = vit_model(masked_img)
        
        assert output is not None
        assert output.shape == (1, 1, 64, 64)
        assert not torch.any(torch.isnan(output))
        assert not torch.any(torch.isinf(output))
    
    def test_vit_different_input_sizes(self, device):
        """Test ViT with different input sizes"""
        input_sizes = [32, 64, 128]
        
        for input_size in input_sizes:
            from src.network.vit import ViT
            model = ViT(input_size=input_size, device=device)
            
            test_input = torch.randn(1, 1, input_size, input_size)
            with torch.no_grad():
                output = model(test_input)
            
            assert output.shape == (1, 1, input_size, input_size)
    
    def test_vit_batch_processing(self, vit_model, sample_batch):
        """Test ViT with batch input"""
        images = sample_batch["images"]
        masks = sample_batch["masks"]
        masked_imgs = images * (1 - masks)
        
        with torch.no_grad():
            output = vit_model(masked_imgs)
        
        assert output.shape == (4, 1, 64, 64)
        assert not torch.any(torch.isnan(output))


class TestSwinTransformer:
    """Test cases for SwinTransformer refine model"""
    
    @pytest.fixture
    def refine_model(self, device):
        from src.network.refine import Refine
        return Refine(input_size=64, device=device)
    
    def test_refine_initialization(self, refine_model, device):
        """Test Refine model initialization"""
        assert refine_model.device == device
        assert refine_model.input_size == 64
        # Adaptive depth calculation: max(2, int(log2(64)) - 4) = max(2, 2) = 2
        assert refine_model.depths == [2, 2, 2, 2]
    
    def test_refine_forward_pass(self, refine_model):
        """Test Refine model forward pass"""
        # Create input: coarse predictions (2 channels) + mask (1 channel)
        coarse_pred = torch.randn(1, 2, 64, 64)
        mask = torch.zeros(1, 1, 64, 64)
        mask[:, :, 16:48, 16:48] = 1.0
        
        # Concatenate inputs
        refine_input = torch.cat([coarse_pred, mask], dim=1)
        
        with torch.no_grad():
            output = refine_model(refine_input)
        
        assert output.shape == (1, 1, 64, 64)
        assert not torch.any(torch.isnan(output))
        assert not torch.any(torch.isinf(output))
    
    def test_refine_different_input_sizes(self, device):
        """Test Refine model with different input sizes"""
        from src.network.refine import Refine
        
        test_cases = [
            (32, [2, 2, 2, 2]),  # max(2, int(log2(32)) - 4) = max(2, 1) = 2
            (64, [2, 2, 2, 2]),  # max(2, int(log2(64)) - 4) = max(2, 2) = 2
            (128, [3, 3, 3, 3]), # max(2, int(log2(128)) - 4) = max(2, 3) = 3
        ]
        
        for input_size, expected_depths in test_cases:
            model = Refine(input_size=input_size, device=device)
            assert model.depths == expected_depths
            
            # Test forward pass
            test_input = torch.randn(1, 3, input_size, input_size)
            with torch.no_grad():
                output = model(test_input)
            
            assert output.shape == (1, 1, input_size, input_size)


class TestInpaintModel:
    """Test cases for main Inpaint model combining ViT + SwinTransformer"""
    
    @pytest.fixture
    def inpaint_model(self, device):
        from src.network.network_pro import Inpaint
        return Inpaint(input_size=64, device=device)
    
    def test_inpaint_initialization(self, inpaint_model, device):
        """Test Inpaint model initialization"""
        assert inpaint_model.device == device
        assert inpaint_model.input_size == 64
        assert hasattr(inpaint_model, 'coarse_stage')
        assert hasattr(inpaint_model, 'refine_stage')
    
    def test_inpaint_forward_pass(self, inpaint_model, sample_image, sample_mask):
        """Test complete inpainting pipeline"""
        # Resize to model input size
        img = torch.nn.functional.interpolate(sample_image, size=(64, 64))
        mask = torch.nn.functional.interpolate(sample_mask, size=(64, 64))
        
        with torch.no_grad():
            output = inpaint_model(img, mask)
        
        assert output.shape == (1, 1, 64, 64)
        assert not torch.any(torch.isnan(output))
        assert not torch.any(torch.isinf(output))
    
    def test_inpaint_preserves_unmasked_regions(self, inpaint_model):
        """Test that inpainting preserves unmasked regions"""
        # Create test image and mask
        img = torch.randn(1, 1, 64, 64)
        mask = torch.zeros(1, 1, 64, 64)
        mask[:, :, 20:40, 20:40] = 1.0  # Only mask center region
        
        with torch.no_grad():
            output = inpaint_model(img, mask)
        
        # Check that unmasked regions are preserved
        unmasked_regions = (mask == 0).float()
        preserved_regions = img * unmasked_regions
        output_unmasked = output * unmasked_regions
        
        # Should be identical in unmasked regions (within floating point precision)
        torch.testing.assert_close(output_unmasked, preserved_regions, rtol=1e-5, atol=1e-6)
    
    def test_inpaint_batch_processing(self, inpaint_model, sample_batch):
        """Test inpainting with batch input"""
        images = sample_batch["images"]
        masks = sample_batch["masks"]
        
        with torch.no_grad():
            output = inpaint_model(images, masks)
        
        assert output.shape == (4, 1, 64, 64)
        assert not torch.any(torch.isnan(output))
        assert not torch.any(torch.isinf(output))
    
    def test_inpaint_gradient_flow(self, inpaint_model, sample_image, sample_mask):
        """Test that gradients flow properly through the model"""
        # Resize inputs
        img = torch.nn.functional.interpolate(sample_image, size=(64, 64))
        mask = torch.nn.functional.interpolate(sample_mask, size=(64, 64))
        
        img.requires_grad_(True)
        
        output = inpaint_model(img, mask)
        loss = output.sum()
        loss.backward()
        
        # Check that gradients exist for input
        assert img.grad is not None
        assert not torch.any(torch.isnan(img.grad))
        
        # Check that model parameters have gradients
        for param in inpaint_model.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert not torch.any(torch.isnan(param.grad))
    
    @pytest.mark.parametrize("input_size", [32, 64, 128])
    def test_inpaint_different_sizes(self, device, input_size):
        """Test inpainting model with different input sizes"""
        from src.network.network_pro import Inpaint
        
        model = Inpaint(input_size=input_size, device=device)
        
        img = torch.randn(1, 1, input_size, input_size)
        mask = torch.zeros(1, 1, input_size, input_size)
        mask_size = input_size // 4
        mask[:, :, mask_size:3*mask_size, mask_size:3*mask_size] = 1.0
        
        with torch.no_grad():
            output = model(img, mask)
        
        assert output.shape == (1, 1, input_size, input_size)
        assert not torch.any(torch.isnan(output))


class TestModelSaving:
    """Test model saving and loading functionality"""
    
    def test_model_state_dict_saving(self, inpaint_model, temp_dir):
        """Test saving and loading model state dict"""
        # Save model
        save_path = temp_dir / "test_model.pth"
        torch.save(inpaint_model.state_dict(), save_path)
        
        # Create new model and load state dict
        from src.network.network_pro import Inpaint
        new_model = Inpaint(input_size=64, device=inpaint_model.device)
        new_model.load_state_dict(torch.load(save_path, map_location='cpu'))
        
        # Compare parameters
        for (name1, param1), (name2, param2) in zip(
            inpaint_model.named_parameters(), new_model.named_parameters()
        ):
            assert name1 == name2
            torch.testing.assert_close(param1, param2)
    
    def test_checkpoint_saving(self, inpaint_model, temp_dir):
        """Test saving and loading full checkpoints"""
        checkpoint = {
            'epoch': 10,
            'model_state_dict': inpaint_model.state_dict(),
            'train_loss': 0.5,
            'val_psnr': 25.0
        }
        
        save_path = temp_dir / "checkpoint.pth"
        torch.save(checkpoint, save_path)
        
        # Load and verify
        loaded = torch.load(save_path, map_location='cpu')
        assert loaded['epoch'] == 10
        assert loaded['train_loss'] == 0.5
        assert loaded['val_psnr'] == 25.0
        assert 'model_state_dict' in loaded