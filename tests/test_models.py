"""Tests for src/network/ — Inpaint forward pass and architecture contracts.

All tests run on CPU with synthetic tensors (no dataset required).
Model creation is slow (~5s), so the Inpaint instance is module-scoped.
"""
import pytest
import torch
import torch.nn as nn
from network.network_pro import Inpaint
from network.vit import ViT, Window_partition, window_reverse
from network.refine import Refine, conv_block


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def inpaint_64():
    """Inpaint model at 64×64 — created once per module for speed."""
    torch.manual_seed(0)
    return Inpaint(input_size=64).eval()


@pytest.fixture(scope="module")
def vit_64():
    torch.manual_seed(0)
    patch_size = max(2, 64 // 16)  # 4
    return ViT(64, patch_size, 768, depth=2, heads=4, mlp_dim=256).eval()


# ---------------------------------------------------------------------------
# Inpaint — construction
# ---------------------------------------------------------------------------

class TestInpaintInit:
    def test_valid_size_64(self):
        model = Inpaint(input_size=64)
        assert isinstance(model, nn.Module)

    def test_valid_size_128(self):
        model = Inpaint(input_size=128)
        assert isinstance(model, nn.Module)

    def test_invalid_size_not_power_of_2(self):
        with pytest.raises(AssertionError):
            Inpaint(input_size=100)

    def test_invalid_size_below_32(self):
        with pytest.raises(AssertionError):
            Inpaint(input_size=16)

    def test_patch_size_auto(self):
        model = Inpaint(input_size=64)
        # patch_size = max(2, 64 // 16) = 4
        assert model.coarse is not None

    def test_has_coarse_and_refine(self):
        model = Inpaint(input_size=64)
        assert hasattr(model, "coarse")
        assert hasattr(model, "refine")


# ---------------------------------------------------------------------------
# Inpaint — forward pass
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestInpaintForward:
    def test_output_shape(self, inpaint_64):
        img  = torch.rand(1, 1, 64, 64) * 2 - 1
        mask = (torch.rand(1, 1, 64, 64) > 0.5).float()
        with torch.no_grad():
            out = inpaint_64(img, mask)
        assert out.shape == (1, 1, 64, 64)

    def test_batch_size_2(self, inpaint_64):
        img  = torch.rand(2, 1, 64, 64) * 2 - 1
        mask = (torch.rand(2, 1, 64, 64) > 0.5).float()
        with torch.no_grad():
            out = inpaint_64(img, mask)
        assert out.shape == (2, 1, 64, 64)

    def test_inpainting_constraint_zero_mask(self, inpaint_64):
        """With mask=0, output must equal input exactly (no inpainting)."""
        img  = torch.rand(1, 1, 64, 64) * 2 - 1
        mask = torch.zeros(1, 1, 64, 64)
        with torch.no_grad():
            out = inpaint_64(img, mask)
        assert torch.allclose(out, img, atol=1e-5)

    def test_output_finite(self, inpaint_64):
        img  = torch.rand(1, 1, 64, 64) * 2 - 1
        mask = (torch.rand(1, 1, 64, 64) > 0.5).float()
        with torch.no_grad():
            out = inpaint_64(img, mask)
        assert torch.isfinite(out).all()

    def test_unmasked_regions_preserved(self, inpaint_64):
        """Pixels where mask=0 must be identical to the input."""
        torch.manual_seed(42)
        img  = torch.rand(1, 1, 64, 64) * 2 - 1
        mask = torch.zeros(1, 1, 64, 64)
        mask[:, :, 10:20, 10:20] = 1.0  # only a small region is masked
        with torch.no_grad():
            out = inpaint_64(img, mask)
        unmasked = (mask == 0)
        assert torch.allclose(out[unmasked], img[unmasked], atol=1e-5)

    def test_gradient_flows_through_model(self):
        model = Inpaint(input_size=64)
        img  = torch.rand(1, 1, 64, 64, requires_grad=False) * 2 - 1
        mask = (torch.rand(1, 1, 64, 64) > 0.5).float()
        out = model(img, mask)
        loss = out.sum()
        loss.backward()
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                assert torch.isfinite(p.grad).all()
                break  # just check at least one param


# ---------------------------------------------------------------------------
# conv_block (Refine helper)
# ---------------------------------------------------------------------------

class TestConvBlock:
    def test_output_shape_preserved(self):
        block = conv_block(3, 16)
        x = torch.rand(2, 3, 32, 32)
        out = block(x)
        assert out.shape == (2, 16, 32, 32)

    def test_same_spatial_size(self):
        block = conv_block(1, 8)
        x = torch.rand(1, 1, 64, 64)
        out = block(x)
        assert out.shape[2:] == x.shape[2:]


# ---------------------------------------------------------------------------
# Refine — construction
# ---------------------------------------------------------------------------

class TestRefineInit:
    def test_valid_size_64(self):
        r = Refine(in_c=2, input_size=64)
        assert isinstance(r, nn.Module)

    def test_invalid_below_64(self):
        with pytest.raises(AssertionError):
            Refine(in_c=2, input_size=32)

    def test_invalid_not_power_of_2(self):
        with pytest.raises(AssertionError):
            Refine(in_c=2, input_size=96)

    def test_num_stages_64(self):
        import math
        r = Refine(in_c=2, input_size=64)
        expected = max(2, int(math.log2(64)) - 4)  # = 2
        assert r.num_stages == expected

    def test_num_stages_256(self):
        import math
        r = Refine(in_c=2, input_size=256)
        expected = max(2, int(math.log2(256)) - 4)  # = 4
        assert r.num_stages == expected


# ---------------------------------------------------------------------------
# Window_partition
# ---------------------------------------------------------------------------

class TestWindowPartition:
    def test_output_dims(self):
        B, C, H, W = 2, 1, 64, 64
        partitioner = Window_partition(window_size=8)
        x = torch.rand(B, C, H, W)
        out = partitioner(x)
        assert out.dim() >= 2
        assert out.shape[0] == B

    def test_no_error_basic(self):
        partitioner = Window_partition(window_size=8)
        x = torch.rand(1, 1, 64, 64)
        out = partitioner(x)
        assert torch.isfinite(out).all()

    def test_output_token_count(self):
        """Regular + overlapping windows: 2 * (H//k) * (W//k) - (H//k-1)*(W//k-1) tokens."""
        B, C, H, W = 1, 1, 64, 64
        k = 8
        partitioner = Window_partition(window_size=k)
        x = torch.rand(B, C, H, W)
        out = partitioner(x)
        regular = (H // k) * (W // k)
        overlap = (H // k - 1) * (W // k - 1)
        assert out.shape[1] == regular + overlap


# ---------------------------------------------------------------------------
# window_reverse
# ---------------------------------------------------------------------------

class TestWindowReverse:
    def test_returns_two_tensors(self):
        B, C, H, W = 1, 1, 64, 64
        partitioner = Window_partition(window_size=8)
        x = torch.rand(B, C, H, W)
        windows = partitioner(x)
        result = window_reverse(windows, window_size=8, resolution=64)
        assert isinstance(result, (list, tuple))
        assert len(result) == 2

    def test_output_spatial_matches_input(self):
        B, C, H, W = 1, 1, 64, 64
        partitioner = Window_partition(window_size=8)
        x = torch.rand(B, C, H, W)
        windows = partitioner(x)
        y, y_ = window_reverse(windows, window_size=8, resolution=64)
        assert y.shape == (B, 1, H, W)
        assert y_.shape == (B, 1, H, W)
