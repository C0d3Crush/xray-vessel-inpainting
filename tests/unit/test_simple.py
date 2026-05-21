import pytest
import torch
import numpy as np


def test_basic_torch():
    """Basic test to verify PyTorch is working"""
    x = torch.randn(2, 3)
    y = torch.randn(2, 3)
    z = x + y
    assert z.shape == (2, 3)
    assert not torch.any(torch.isnan(z))


def test_basic_numpy():
    """Basic test to verify NumPy is working"""
    x = np.random.randn(2, 3)
    y = np.random.randn(2, 3)
    z = x + y
    assert z.shape == (2, 3)
    assert not np.any(np.isnan(z))


def test_device_availability():
    """Test device availability"""
    device = torch.device('cpu')
    assert device.type == 'cpu'
    
    if torch.cuda.is_available():
        cuda_device = torch.device('cuda')
        assert cuda_device.type == 'cuda'


@pytest.mark.parametrize("input_size", [32, 64, 128])
def test_tensor_operations(input_size):
    """Test basic tensor operations with different sizes"""
    x = torch.randn(1, 1, input_size, input_size)
    assert x.shape == (1, 1, input_size, input_size)
    
    # Test basic operations
    mean_val = torch.mean(x)
    assert not torch.isnan(mean_val)
    assert not torch.isinf(mean_val)


class TestBasicFunctionality:
    """Test class for basic functionality"""
    
    def test_fixture_usage(self, temp_dir):
        """Test that fixtures work correctly"""
        assert temp_dir.exists()
        assert temp_dir.is_dir()
        
        # Create a test file
        test_file = temp_dir / "test.txt"
        test_file.write_text("Hello, testing!")
        
        assert test_file.exists()
        assert test_file.read_text() == "Hello, testing!"
    
    def test_mock_functionality(self, mock_coco_annotation):
        """Test that mock data fixtures work"""
        assert "images" in mock_coco_annotation
        assert "annotations" in mock_coco_annotation
        assert "categories" in mock_coco_annotation
        
        assert len(mock_coco_annotation["images"]) == 1
        assert len(mock_coco_annotation["annotations"]) == 1
        assert len(mock_coco_annotation["categories"]) == 1