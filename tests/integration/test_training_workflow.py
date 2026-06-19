import sys
import os
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from trainer import train_model
from network.network_pro import Inpaint

COMMON_ARGS = dict(
    epochs=1,
    batch_size=1,
    input_size=64,
    patches_per_image=1,
    device='cpu',
    seed=42,
    num_workers=0,
)


@pytest.mark.slow
@pytest.mark.integration
def test_single_epoch_completes(tmp_path, mock_dataset_dir):
    result = train_model(
        train_img=mock_dataset_dir["img_dir"],
        train_ann=mock_dataset_dir["ann_path"],
        val_img=mock_dataset_dir["img_dir"],
        val_ann=mock_dataset_dir["ann_path"],
        output_dir=str(tmp_path / "checkpoints"),
        **COMMON_ARGS,
    )
    assert isinstance(result, dict)
    assert result['best_val_psnr'] >= 0
    assert (tmp_path / "checkpoints" / "best.pth").exists()
    assert (tmp_path / "checkpoints" / "training_log.csv").exists()


@pytest.mark.slow
@pytest.mark.integration
def test_resume_from_checkpoint(tmp_path, mock_dataset_dir):
    ckpt_dir = str(tmp_path / "checkpoints")
    shared = dict(
        train_img=mock_dataset_dir["img_dir"],
        train_ann=mock_dataset_dir["ann_path"],
        val_img=mock_dataset_dir["img_dir"],
        val_ann=mock_dataset_dir["ann_path"],
        output_dir=ckpt_dir,
        **COMMON_ARGS,
    )
    train_model(**shared)

    ckpt_path = str(tmp_path / "checkpoints" / "best.pth")
    result2 = train_model(**{**shared, 'epochs': 2, 'ckpt': ckpt_path})
    assert isinstance(result2['best_val_psnr'], float)


@pytest.mark.integration
def test_model_output_in_range():
    model = Inpaint(input_size=64).eval()
    img = torch.randn(1, 1, 64, 64).clamp(-1.0, 1.0)
    mask = torch.randint(0, 2, (1, 1, 64, 64)).float()
    with torch.no_grad():
        out = model(img, mask)
    assert out.min().item() >= -1.0, f"Output below -1: {out.min().item()}"
    assert out.max().item() <= 1.0, f"Output above 1: {out.max().item()}"
