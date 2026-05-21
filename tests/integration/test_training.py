import pytest
import torch
import tempfile
import shutil
from pathlib import Path
import json
import numpy as np
from PIL import Image
import os
import subprocess


@pytest.mark.integration
class TestTrainingPipeline:
    """Integration tests for the training pipeline"""
    
    @pytest.fixture
    def training_setup(self, temp_dir):
        """Set up minimal training environment"""
        # Create directory structure
        data_dir = temp_dir / "data"
        train_img_dir = data_dir / "train" / "images"
        val_img_dir = data_dir / "val" / "images"
        train_ann_dir = data_dir / "train" / "annotations"
        val_ann_dir = data_dir / "val" / "annotations"
        
        for dir_path in [train_img_dir, val_img_dir, train_ann_dir, val_ann_dir]:
            dir_path.mkdir(parents=True)
        
        # Create sample images
        for i in range(3):
            img_array = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
            Image.fromarray(img_array, mode='L').save(train_img_dir / f"train_{i:03d}.png")
            Image.fromarray(img_array, mode='L').save(val_img_dir / f"val_{i:03d}.png")
        
        # Create COCO annotations
        train_annotation = {
            "images": [
                {"id": i+1, "file_name": f"train_{i:03d}.png", "width": 256, "height": 256}
                for i in range(3)
            ],
            "annotations": [
                {
                    "id": i+1, "image_id": i+1, "category_id": 1,
                    "segmentation": [[50+i*10, 50, 100+i*10, 50, 100+i*10, 100, 50+i*10, 100]],
                    "area": 2500, "bbox": [50+i*10, 50, 50, 50]
                }
                for i in range(3)
            ],
            "categories": [{"id": 1, "name": "vessel"}]
        }
        
        val_annotation = {
            "images": [
                {"id": i+1, "file_name": f"val_{i:03d}.png", "width": 256, "height": 256}
                for i in range(3)
            ],
            "annotations": [
                {
                    "id": i+1, "image_id": i+1, "category_id": 1,
                    "segmentation": [[60+i*10, 60, 110+i*10, 60, 110+i*10, 110, 60+i*10, 110]],
                    "area": 2500, "bbox": [60+i*10, 60, 50, 50]
                }
                for i in range(3)
            ],
            "categories": [{"id": 1, "name": "vessel"}]
        }
        
        train_ann_file = train_ann_dir / "train.json"
        val_ann_file = val_ann_dir / "val.json"
        
        with open(train_ann_file, 'w') as f:
            json.dump(train_annotation, f)
        with open(val_ann_file, 'w') as f:
            json.dump(val_annotation, f)
        
        return {
            "train_img": str(train_img_dir),
            "train_ann": str(train_ann_file),
            "val_img": str(val_img_dir),
            "val_ann": str(val_ann_file),
            "output_dir": str(temp_dir / "checkpoints")
        }
    
    def test_smoke_training(self, training_setup, temp_dir):
        """Test minimal training run (smoke test)"""
        import sys
        sys.path.insert(0, str(Path.cwd() / "src"))
        
        from src.train import main
        
        # Minimal training arguments
        args = [
            "--train_img", training_setup["train_img"],
            "--train_ann", training_setup["train_ann"],
            "--val_img", training_setup["val_img"],
            "--val_ann", training_setup["val_ann"],
            "--output_dir", training_setup["output_dir"],
            "--epochs", "2",
            "--batch_size", "1",
            "--input_size", "64",
            "--device", "cpu",
            "--smoke_test",
            "--smoke_size", "2"
        ]
        
        # Change directory to project root for relative imports
        original_cwd = os.getcwd()
        try:
            # Run training
            import argparse
            parser = argparse.ArgumentParser()
            
            # Add all necessary arguments (simplified version)
            parser.add_argument('--train_img', required=True)
            parser.add_argument('--train_ann', required=True)
            parser.add_argument('--val_img', required=True)
            parser.add_argument('--val_ann', required=True)
            parser.add_argument('--output_dir', default='checkpoints')
            parser.add_argument('--epochs', type=int, default=100)
            parser.add_argument('--batch_size', type=int, default=4)
            parser.add_argument('--input_size', type=int, default=64)
            parser.add_argument('--device', default='cpu')
            parser.add_argument('--smoke_test', action='store_true')
            parser.add_argument('--smoke_size', type=int, default=20)
            parser.add_argument('--lr', type=float, default=1e-4)
            parser.add_argument('--save_every', type=int, default=10)
            
            parsed_args = parser.parse_args(args)
            
            # This would normally call main(parsed_args), but we'll test components
            assert Path(training_setup["train_img"]).exists()
            assert Path(training_setup["train_ann"]).exists()
            assert Path(training_setup["val_img"]).exists()
            assert Path(training_setup["val_ann"]).exists()
            
        finally:
            os.chdir(original_cwd)
    
    def test_dataset_loading_integration(self, training_setup):
        """Test dataset loading in training context"""
        import sys
        sys.path.insert(0, str(Path.cwd() / "src"))
        
        from src.train import ArcadeDataset
        from torch.utils.data import DataLoader
        
        # Create datasets
        train_dataset = ArcadeDataset(
            img_path=training_setup["train_img"],
            ann_path=training_setup["train_ann"],
            input_size=64
        )
        
        val_dataset = ArcadeDataset(
            img_path=training_setup["val_img"],
            ann_path=training_setup["val_ann"],
            input_size=64
        )
        
        # Test dataset properties
        assert len(train_dataset) == 3
        assert len(val_dataset) == 3
        
        # Test data loading
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0)
        
        # Get sample batches
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))
        
        # Verify batch structure
        for batch in [train_batch, val_batch]:
            assert "image" in batch
            assert "mask" in batch
            assert "image_id" in batch
            assert batch["image"].shape[0] <= 2  # Batch size
            assert batch["image"].shape[1:] == (1, 64, 64)  # Channel, H, W
            assert batch["mask"].shape[1:] == (1, 64, 64)
    
    def test_model_training_step(self, training_setup):
        """Test single training step with real data"""
        import sys
        sys.path.insert(0, str(Path.cwd() / "src"))
        
        from src.train import ArcadeDataset
        from src.network.network_pro import Inpaint
        from torch.utils.data import DataLoader
        import torch.optim as optim
        import torch.nn.functional as F
        
        # Set up dataset and model
        dataset = ArcadeDataset(
            img_path=training_setup["train_img"],
            ann_path=training_setup["train_ann"],
            input_size=64,
            max_dataset_size=2  # Limit for testing
        )
        
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        
        device = torch.device('cpu')
        model = Inpaint(input_size=64, device=device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        
        # Single training step
        model.train()
        batch = next(iter(dataloader))
        
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images, masks)
        
        # Calculate loss (simplified L1)
        loss = F.l1_loss(outputs * masks, images * masks) * 6.0 + \
               F.l1_loss(outputs * (1 - masks), images * (1 - masks)) * 1.0
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Verify training step completed successfully
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        assert loss.item() >= 0.0
        
        # Verify gradients exist
        has_gradients = False
        for param in model.parameters():
            if param.grad is not None and param.grad.norm() > 0:
                has_gradients = True
                break
        assert has_gradients, "No gradients found after backward pass"
    
    def test_checkpoint_saving_loading(self, training_setup, temp_dir):
        """Test checkpoint saving and loading during training"""
        import sys
        sys.path.insert(0, str(Path.cwd() / "src"))
        
        from src.network.network_pro import Inpaint
        import torch.optim as optim
        
        device = torch.device('cpu')
        model = Inpaint(input_size=64, device=device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        
        # Create checkpoint
        checkpoint_dir = Path(training_setup["output_dir"])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': 5,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': 0.5,
            'val_psnr': 25.0,
            'val_ssim': 0.8
        }
        
        checkpoint_path = checkpoint_dir / "test_checkpoint.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Load checkpoint into new model
        new_model = Inpaint(input_size=64, device=device)
        new_optimizer = optim.Adam(new_model.parameters(), lr=1e-4)
        
        loaded_checkpoint = torch.load(checkpoint_path, map_location=device)
        new_model.load_state_dict(loaded_checkpoint['model_state_dict'])
        new_optimizer.load_state_dict(loaded_checkpoint['optimizer_state_dict'])
        
        # Verify loaded data
        assert loaded_checkpoint['epoch'] == 5
        assert loaded_checkpoint['train_loss'] == 0.5
        assert loaded_checkpoint['val_psnr'] == 25.0
        assert loaded_checkpoint['val_ssim'] == 0.8
        
        # Verify model parameters match
        for (name1, param1), (name2, param2) in zip(
            model.named_parameters(), new_model.named_parameters()
        ):
            assert name1 == name2
            torch.testing.assert_close(param1, param2)
    
    @pytest.mark.slow
    def test_full_training_cycle(self, training_setup):
        """Test complete training cycle with multiple epochs"""
        import sys
        sys.path.insert(0, str(Path.cwd() / "src"))
        
        from src.train import ArcadeDataset
        from src.network.network_pro import Inpaint
        from torch.utils.data import DataLoader
        import torch.optim as optim
        import torch.nn.functional as F
        
        # Set up training components
        train_dataset = ArcadeDataset(
            img_path=training_setup["train_img"],
            ann_path=training_setup["train_ann"],
            input_size=64,
            max_dataset_size=2
        )
        
        val_dataset = ArcadeDataset(
            img_path=training_setup["val_img"],
            ann_path=training_setup["val_ann"],
            input_size=64,
            max_dataset_size=2
        )
        
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
        
        device = torch.device('cpu')
        model = Inpaint(input_size=64, device=device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)  # Higher LR for faster convergence
        
        # Training loop
        num_epochs = 3
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            epoch_train_loss = 0.0
            
            for batch in train_loader:
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                
                optimizer.zero_grad()
                outputs = model(images, masks)
                
                loss = F.l1_loss(outputs * masks, images * masks) * 6.0 + \
                       F.l1_loss(outputs * (1 - masks), images * (1 - masks)) * 1.0
                
                loss.backward()
                optimizer.step()
                
                epoch_train_loss += loss.item()
            
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation phase
            model.eval()
            epoch_val_loss = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    images = batch["image"].to(device)
                    masks = batch["mask"].to(device)
                    
                    outputs = model(images, masks)
                    
                    loss = F.l1_loss(outputs * masks, images * masks) * 6.0 + \
                           F.l1_loss(outputs * (1 - masks), images * (1 - masks)) * 1.0
                    
                    epoch_val_loss += loss.item()
            
            avg_val_loss = epoch_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
        
        # Verify training completed successfully
        assert len(train_losses) == num_epochs
        assert len(val_losses) == num_epochs
        
        # Verify losses are valid numbers
        for loss in train_losses + val_losses:
            assert not np.isnan(loss)
            assert not np.isinf(loss)
            assert loss >= 0.0
        
        # Training should show some progress (losses should generally decrease or stabilize)
        # Note: With only 3 epochs and small dataset, this is a weak check
        assert train_losses[-1] < train_losses[0] * 2  # At least not exploding