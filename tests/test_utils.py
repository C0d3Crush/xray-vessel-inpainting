"""Tests for src/utils.py — metrics and checkpoint helpers."""
import json
import os
import pytest
import numpy as np
import torch
import torch.nn as nn
from utils import (
    psnr, rmse, wasserstein_distance_2d, calculate_kl_divergence,
    save_checkpoint, load_checkpoint, rotate_checkpoints,
    load_coco_annotations, STENOSIS_CATEGORY_ID,
)


# ---------------------------------------------------------------------------
# Tiny model for checkpoint tests
# ---------------------------------------------------------------------------

class _TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


# ---------------------------------------------------------------------------
# psnr
# ---------------------------------------------------------------------------

class TestPsnr:
    def test_identical_images_returns_100(self):
        img = np.ones((64, 64), dtype=np.float32) * 128
        assert psnr(img, img) == 100

    def test_zero_mse_returns_100(self):
        a = np.zeros((32, 32))
        assert psnr(a, a) == 100

    def test_returns_float(self):
        a = np.random.rand(32, 32) * 255
        b = np.random.rand(32, 32) * 255
        assert isinstance(psnr(a, b), float)

    def test_higher_difference_lower_psnr(self):
        ref = np.ones((64, 64)) * 128.0
        close = ref + 1.0
        far   = ref + 50.0
        assert psnr(ref, close) > psnr(ref, far)

    def test_positive_value(self):
        a = np.random.rand(64, 64) * 255
        b = np.random.rand(64, 64) * 255
        assert psnr(a, b) > 0

    def test_formula(self):
        a = np.zeros((10, 10))
        b = np.full((10, 10), 10.0)
        expected = 20 * np.log10(255.0 / np.sqrt(np.mean((a - b) ** 2)))
        assert abs(psnr(a, b) - expected) < 1e-9


# ---------------------------------------------------------------------------
# rmse
# ---------------------------------------------------------------------------

class TestRmse:
    def test_identical_images_returns_zero(self):
        a = np.ones((32, 32)) * 100
        assert rmse(a, a) == 0.0

    def test_returns_float(self):
        a = np.random.rand(32, 32) * 255
        b = np.random.rand(32, 32) * 255
        assert isinstance(rmse(a, b), float)

    def test_non_negative(self):
        a = np.random.rand(64, 64) * 255
        b = np.random.rand(64, 64) * 255
        assert rmse(a, b) >= 0

    def test_formula(self):
        a = np.array([[0.0, 0.0], [0.0, 0.0]])
        b = np.array([[3.0, 4.0], [0.0, 0.0]])
        expected = np.sqrt(np.mean([9, 16, 0, 0]))
        assert abs(rmse(a, b) - expected) < 1e-9

    def test_higher_difference_higher_rmse(self):
        ref = np.zeros((64, 64))
        assert rmse(ref, ref + 1) < rmse(ref, ref + 10)

    def test_any_shape(self):
        a = np.random.rand(4, 4, 3) * 255
        b = np.random.rand(4, 4, 3) * 255
        result = rmse(a, b)
        assert result >= 0


# ---------------------------------------------------------------------------
# wasserstein_distance_2d
# ---------------------------------------------------------------------------

class TestWassersteinDistance2d:
    def test_identical_near_zero(self):
        a = np.random.rand(64, 64) * 255
        assert wasserstein_distance_2d(a, a) < 1e-6

    def test_non_negative(self):
        a = np.random.rand(32, 32) * 255
        b = np.random.rand(32, 32) * 255
        assert wasserstein_distance_2d(a, b) >= 0

    def test_returns_float(self):
        a = np.random.rand(32, 32) * 255
        b = np.random.rand(32, 32) * 255
        assert isinstance(wasserstein_distance_2d(a, b), float)

    def test_larger_shift_larger_distance(self):
        base = np.zeros((64, 64))
        shifted_small = base + 10
        shifted_large = base + 100
        assert wasserstein_distance_2d(base, shifted_small) < wasserstein_distance_2d(base, shifted_large)


# ---------------------------------------------------------------------------
# calculate_kl_divergence
# ---------------------------------------------------------------------------

class TestCalculateKlDivergence:
    def test_identical_near_zero(self):
        a = np.random.rand(64, 64)
        kl = calculate_kl_divergence(a, a)
        assert kl < 0.1  # Not exactly 0 due to histogram binning + epsilon

    def test_returns_float(self):
        a = np.random.rand(32, 32)
        b = np.random.rand(32, 32)
        assert isinstance(calculate_kl_divergence(a, b), float)

    def test_non_negative(self):
        a = np.random.rand(64, 64)
        b = np.random.rand(64, 64)
        assert calculate_kl_divergence(a, b) >= 0

    def test_accepts_torch_tensor(self):
        a = torch.rand(32, 32)
        b = torch.rand(32, 32)
        result = calculate_kl_divergence(a, b)
        assert isinstance(result, float)

    def test_accepts_mixed_types(self):
        a = np.random.rand(32, 32)
        b = torch.rand(32, 32)
        result = calculate_kl_divergence(a, b)
        assert isinstance(result, float)

    def test_different_distributions_higher_kl(self):
        uniform = np.ones((64, 64))
        bimodal = np.zeros((64, 64))
        bimodal[:32, :] = 1.0
        bimodal[32:, :] = 200.0
        kl_same = calculate_kl_divergence(uniform, uniform)
        kl_diff = calculate_kl_divergence(uniform, bimodal)
        assert kl_diff > kl_same


# ---------------------------------------------------------------------------
# save_checkpoint / load_checkpoint
# ---------------------------------------------------------------------------

class TestCheckpointRoundtrip:
    def test_save_creates_file(self, tmp_path):
        model = _TinyNet()
        opt = torch.optim.Adam(model.parameters())
        path = str(tmp_path / "ckpt.pth")
        save_checkpoint(model, opt, epoch=1, path=path)
        assert os.path.exists(path)

    def test_save_returns_dict(self, tmp_path):
        model = _TinyNet()
        opt = torch.optim.Adam(model.parameters())
        path = str(tmp_path / "ckpt.pth")
        ckpt = save_checkpoint(model, opt, epoch=5, path=path)
        assert "state_dict" in ckpt
        assert "optimizer" in ckpt
        assert ckpt["epoch"] == 5

    def test_save_with_metrics(self, tmp_path):
        model = _TinyNet()
        opt = torch.optim.Adam(model.parameters())
        path = str(tmp_path / "ckpt.pth")
        ckpt = save_checkpoint(model, opt, epoch=1, path=path, metrics={"val_psnr": 32.5})
        assert ckpt["val_psnr"] == 32.5

    def test_load_restores_weights(self, tmp_path):
        model = _TinyNet()
        torch.nn.init.constant_(model.fc.weight, 0.42)
        torch.nn.init.constant_(model.fc.bias, 0.0)
        opt = torch.optim.Adam(model.parameters())
        path = str(tmp_path / "ckpt.pth")
        save_checkpoint(model, opt, epoch=1, path=path)

        model2 = _TinyNet()
        torch.nn.init.zeros_(model2.fc.weight)
        model2 = load_checkpoint(path, model2, "cpu")
        assert torch.allclose(model2.fc.weight, torch.full((2, 4), 0.42))

    def test_load_missing_file_raises(self, tmp_path):
        model = _TinyNet()
        with pytest.raises(Exception):
            load_checkpoint(str(tmp_path / "nonexistent.pth"), model, "cpu")

    def test_save_no_optimizer(self, tmp_path):
        model = _TinyNet()
        path = str(tmp_path / "ckpt.pth")
        ckpt = save_checkpoint(model, None, epoch=1, path=path)
        assert ckpt["optimizer"] is None


# ---------------------------------------------------------------------------
# rotate_checkpoints
# ---------------------------------------------------------------------------

class TestRotateCheckpoints:
    def _make_epoch_ckpts(self, directory, epochs):
        """Create dummy epoch_*.pth files with different mtimes."""
        import time
        paths = []
        for e in epochs:
            p = os.path.join(directory, f"epoch_{e:03d}.pth")
            torch.save({}, p)
            time.sleep(0.01)  # ensure different mtimes
            paths.append(p)
        return paths

    def test_keeps_n_most_recent(self, tmp_path):
        d = str(tmp_path)
        self._make_epoch_ckpts(d, range(1, 6))  # epoch_001 .. epoch_005
        rotate_checkpoints(d, keep_checkpoints=3)
        remaining = [f for f in os.listdir(d) if f.startswith("epoch_")]
        assert len(remaining) == 3

    def test_preserves_best_pth(self, tmp_path):
        d = str(tmp_path)
        best = tmp_path / "best.pth"
        torch.save({}, best)
        self._make_epoch_ckpts(d, range(1, 6))
        rotate_checkpoints(d, keep_checkpoints=2)
        assert best.exists()

    def test_no_op_when_keep_zero(self, tmp_path):
        d = str(tmp_path)
        self._make_epoch_ckpts(d, range(1, 4))
        rotate_checkpoints(d, keep_checkpoints=0)
        remaining = [f for f in os.listdir(d) if f.startswith("epoch_")]
        assert len(remaining) == 3  # nothing removed

    def test_no_op_when_under_limit(self, tmp_path):
        d = str(tmp_path)
        self._make_epoch_ckpts(d, range(1, 3))  # 2 files
        rotate_checkpoints(d, keep_checkpoints=5)
        remaining = [f for f in os.listdir(d) if f.startswith("epoch_")]
        assert len(remaining) == 2

    def test_empty_directory_no_error(self, tmp_path):
        rotate_checkpoints(str(tmp_path), keep_checkpoints=3)  # should not raise


# ---------------------------------------------------------------------------
# load_coco_annotations
# ---------------------------------------------------------------------------

class TestLoadCocoAnnotations:
    @pytest.fixture
    def coco_file(self, tmp_path):
        data = {
            "images": [
                {"id": 1, "file_name": "a.png", "width": 64, "height": 64},
                {"id": 2, "file_name": "b.png", "width": 64, "height": 64},
                {"id": 3, "file_name": "c.png", "width": 64, "height": 64},
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1,
                 "segmentation": [[0, 0, 10, 0, 10, 10, 0, 10]]},
                {"id": 2, "image_id": 1, "category_id": 26,  # stenosis — must be excluded
                 "segmentation": [[5, 5, 15, 5, 15, 15, 5, 15]]},
                {"id": 3, "image_id": 2, "category_id": 1,
                 "segmentation": [[20, 20, 30, 20, 30, 30, 20, 30]]},
                # image 3 has no annotations → excluded from image_ids
            ],
            "categories": [{"id": 1, "name": "vessel"}, {"id": 26, "name": "stenosis"}],
        }
        path = tmp_path / "ann.json"
        path.write_text(json.dumps(data))
        return str(path)

    def test_returns_three_values(self, coco_file):
        result = load_coco_annotations(coco_file)
        assert len(result) == 3

    def test_id_to_info_has_all_images(self, coco_file):
        id_to_info, _, _ = load_coco_annotations(coco_file)
        assert set(id_to_info.keys()) == {1, 2, 3}

    def test_stenosis_excluded(self, coco_file):
        _, anns_by_image, _ = load_coco_annotations(coco_file)
        for anns in anns_by_image.values():
            for ann in anns:
                assert ann['category_id'] != STENOSIS_CATEGORY_ID

    def test_image_ids_only_annotated(self, coco_file):
        _, _, image_ids = load_coco_annotations(coco_file)
        assert 3 not in image_ids  # image 3 has no vessel annotations

    def test_image_ids_contains_annotated(self, coco_file):
        _, _, image_ids = load_coco_annotations(coco_file)
        assert 1 in image_ids
        assert 2 in image_ids

    def test_stenosis_only_image_excluded_from_ids(self, tmp_path):
        data = {
            "images": [{"id": 1, "file_name": "a.png", "width": 64, "height": 64}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 26,
                 "segmentation": [[0, 0, 10, 0, 10, 10]]},
            ],
            "categories": [],
        }
        path = tmp_path / "ann.json"
        path.write_text(json.dumps(data))
        _, _, image_ids = load_coco_annotations(str(path))
        assert 1 not in image_ids  # only stenosis → no valid annotations
