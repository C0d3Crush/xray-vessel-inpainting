"""Tests for src/losses.py — InpaintingLoss and SSIM helpers."""
import pytest
import torch
from losses import InpaintingLoss, _build_ssim_window


# ---------------------------------------------------------------------------
# _build_ssim_window
# ---------------------------------------------------------------------------

class TestBuildSsimWindow:
    def test_output_shape(self):
        w = _build_ssim_window(11, torch.float32, torch.device("cpu"))
        assert w.shape == (1, 1, 11, 11)

    def test_sums_to_one(self):
        w = _build_ssim_window(11, torch.float32, torch.device("cpu"))
        assert abs(w.sum().item() - 1.0) < 1e-5

    def test_non_negative(self):
        w = _build_ssim_window(11, torch.float32, torch.device("cpu"))
        assert (w >= 0).all()

    def test_symmetric(self):
        w = _build_ssim_window(11, torch.float32, torch.device("cpu"))
        kernel = w.squeeze()
        assert torch.allclose(kernel, kernel.T, atol=1e-6)

    def test_window_size_3(self):
        w = _build_ssim_window(3, torch.float32, torch.device("cpu"))
        assert w.shape == (1, 1, 3, 3)
        assert abs(w.sum().item() - 1.0) < 1e-5

    def test_dtype_preserved(self):
        w = _build_ssim_window(7, torch.float64, torch.device("cpu"))
        assert w.dtype == torch.float64


# ---------------------------------------------------------------------------
# InpaintingLoss — construction
# ---------------------------------------------------------------------------

class TestInpaintingLossInit:
    def test_defaults(self):
        loss = InpaintingLoss()
        assert loss.ssim_weight == 0.5
        assert loss.mask_weight == 6.0
        assert loss.valid_weight == 1.0
        assert loss.ssim_window_size == 11
        assert loss._ssim_window is None  # lazy

    def test_custom_weights(self):
        loss = InpaintingLoss(ssim_weight=1.0, mask_weight=3.0, valid_weight=2.0)
        assert loss.ssim_weight == 1.0
        assert loss.mask_weight == 3.0
        assert loss.valid_weight == 2.0


# ---------------------------------------------------------------------------
# InpaintingLoss.forward
# ---------------------------------------------------------------------------

class TestInpaintingLossForward:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.loss_fn = InpaintingLoss()
        self.B, self.H, self.W = 2, 64, 64
        torch.manual_seed(0)
        self.output = torch.rand(self.B, 1, self.H, self.W) * 2 - 1
        self.target = torch.rand(self.B, 1, self.H, self.W) * 2 - 1
        self.mask   = (torch.rand(self.B, 1, self.H, self.W) > 0.5).float()

    def test_return_types(self):
        total, components = self.loss_fn(self.output, self.target, self.mask)
        assert isinstance(total, torch.Tensor)
        assert isinstance(components, dict)

    def test_total_loss_is_scalar(self):
        total, _ = self.loss_fn(self.output, self.target, self.mask)
        assert total.shape == torch.Size([])

    def test_component_keys(self):
        _, components = self.loss_fn(self.output, self.target, self.mask)
        assert set(components.keys()) == {"l1_loss", "ssim_loss", "mask_loss", "valid_loss"}

    def test_all_finite(self):
        total, components = self.loss_fn(self.output, self.target, self.mask)
        assert torch.isfinite(total)
        for v in components.values():
            assert torch.isfinite(torch.tensor(v))

    def test_total_positive(self):
        total, _ = self.loss_fn(self.output, self.target, self.mask)
        assert total.item() > 0

    def test_identical_inputs_low_loss(self):
        x = torch.rand(1, 1, 64, 64) * 2 - 1
        mask = torch.zeros(1, 1, 64, 64)
        total, _ = self.loss_fn(x, x, mask)
        # With identical output and target, L1 terms are 0; only SSIM remains near 0
        assert total.item() < 0.1

    def test_all_mask_zeros_mask_loss_near_zero(self):
        """When mask=0, the masked region L1 should be ~0."""
        mask = torch.zeros(self.B, 1, self.H, self.W)
        _, components = self.loss_fn(self.output, self.target, mask)
        assert components["mask_loss"] < 1e-6

    def test_all_mask_ones_valid_loss_near_zero(self):
        """When mask=1, the valid region L1 should be ~0."""
        mask = torch.ones(self.B, 1, self.H, self.W)
        _, components = self.loss_fn(self.output, self.target, mask)
        assert components["valid_loss"] < 1e-6

    def test_gradient_flows(self):
        output = self.output.requires_grad_(True)
        total, _ = self.loss_fn(output, self.target, self.mask)
        total.backward()
        assert output.grad is not None
        assert not torch.isnan(output.grad).any()

    def test_ssim_window_cached_after_first_call(self):
        assert self.loss_fn._ssim_window is None
        self.loss_fn(self.output, self.target, self.mask)
        assert self.loss_fn._ssim_window is not None

    def test_loss_increases_with_difference(self):
        x = torch.zeros(1, 1, 64, 64)
        y_close = x + 0.01
        y_far   = x + 1.0
        mask = torch.ones(1, 1, 64, 64)
        loss_close, _ = self.loss_fn(x, y_close, mask)
        loss_far,   _ = self.loss_fn(x, y_far,   mask)
        assert loss_far.item() > loss_close.item()

    def test_weighted_sum_matches_total(self):
        total, components = self.loss_fn(self.output, self.target, self.mask)
        expected = (
            components["mask_loss"] * self.loss_fn.mask_weight
            + components["valid_loss"] * self.loss_fn.valid_weight
            + components["ssim_loss"]  # already scaled in components
        )
        assert abs(total.item() - expected) < 1e-4

    def test_batch_size_1(self):
        out = torch.rand(1, 1, 64, 64) * 2 - 1
        tgt = torch.rand(1, 1, 64, 64) * 2 - 1
        msk = torch.ones(1, 1, 64, 64)
        total, _ = self.loss_fn(out, tgt, msk)
        assert torch.isfinite(total)


# ---------------------------------------------------------------------------
# InpaintingLoss._ssim_loss
# ---------------------------------------------------------------------------

class TestSsimLoss:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.loss_fn = InpaintingLoss()

    def test_identical_tensors_near_zero(self):
        x = torch.rand(1, 1, 64, 64)
        loss = self.loss_fn._ssim_loss(x, x)
        assert loss.item() < 0.01

    def test_different_tensors_positive(self):
        x = torch.zeros(1, 1, 64, 64)
        y = torch.ones(1, 1, 64, 64)
        loss = self.loss_fn._ssim_loss(x, y)
        assert loss.item() > 0

    def test_scalar_output(self):
        x = torch.rand(2, 1, 64, 64)
        loss = self.loss_fn._ssim_loss(x, x)
        assert loss.shape == torch.Size([])
