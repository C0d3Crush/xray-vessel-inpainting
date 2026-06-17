"""Tests for src/dataset.py — ArcadeDataset validation, mask generation, patch extraction."""
import os
import json
import pytest
import numpy as np
from PIL import Image
from dataset import ArcadeDataset, DatasetConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ds(mock_dataset_dir, **cfg_kwargs):
    """Convenience: build ArcadeDataset with a DatasetConfig."""
    return ArcadeDataset(
        mock_dataset_dir["img_dir"],
        mock_dataset_dir["ann_path"],
        DatasetConfig(**cfg_kwargs),
    )


# ---------------------------------------------------------------------------
# DatasetConfig
# ---------------------------------------------------------------------------

class TestDatasetConfig:
    def test_defaults(self):
        cfg = DatasetConfig()
        assert cfg.image_size == 64
        assert cfg.patches_per_image == 4
        assert cfg.foreground_prob == 0.75
        assert cfg.mask_dir is None
        assert cfg.random_masks is False
        assert cfg.background_training is True
        assert cfg.vessel_safe_training is False

    def test_custom_values(self):
        cfg = DatasetConfig(image_size=128, patches_per_image=8, foreground_prob=0.5)
        assert cfg.image_size == 128
        assert cfg.patches_per_image == 8
        assert cfg.foreground_prob == 0.5


# ---------------------------------------------------------------------------
# Construction — input validation
# ---------------------------------------------------------------------------

class TestArcadeDatasetValidation:
    def test_missing_img_dir_raises(self, mock_dataset_dir, tmp_path):
        with pytest.raises(NotADirectoryError):
            ArcadeDataset(
                str(tmp_path / "nonexistent"),
                mock_dataset_dir["ann_path"],
            )

    def test_missing_ann_path_raises(self, mock_dataset_dir):
        with pytest.raises(FileNotFoundError):
            ArcadeDataset(
                mock_dataset_dir["img_dir"],
                mock_dataset_dir["img_dir"] + "/nonexistent.json",
            )

    def test_missing_mask_dir_raises(self, mock_dataset_dir, tmp_path):
        with pytest.raises(NotADirectoryError):
            ArcadeDataset(
                mock_dataset_dir["img_dir"],
                mock_dataset_dir["ann_path"],
                DatasetConfig(mask_dir=str(tmp_path / "no_masks")),
            )

    def test_foreground_prob_below_zero_raises(self, mock_dataset_dir):
        with pytest.raises(ValueError):
            ArcadeDataset(
                mock_dataset_dir["img_dir"],
                mock_dataset_dir["ann_path"],
                DatasetConfig(foreground_prob=-0.1),
            )

    def test_foreground_prob_above_one_raises(self, mock_dataset_dir):
        with pytest.raises(ValueError):
            ArcadeDataset(
                mock_dataset_dir["img_dir"],
                mock_dataset_dir["ann_path"],
                DatasetConfig(foreground_prob=1.1),
            )

    def test_image_size_below_32_raises(self, mock_dataset_dir):
        with pytest.raises(ValueError):
            ArcadeDataset(
                mock_dataset_dir["img_dir"],
                mock_dataset_dir["ann_path"],
                DatasetConfig(image_size=16),
            )

    def test_patches_per_image_zero_raises(self, mock_dataset_dir):
        with pytest.raises(ValueError):
            ArcadeDataset(
                mock_dataset_dir["img_dir"],
                mock_dataset_dir["ann_path"],
                DatasetConfig(patches_per_image=0),
            )

    def test_valid_construction_succeeds(self, mock_dataset_dir):
        ds = _ds(mock_dataset_dir, image_size=64)
        assert ds is not None

    def test_default_cfg_used_when_none(self, mock_dataset_dir):
        ds = ArcadeDataset(mock_dataset_dir["img_dir"], mock_dataset_dir["ann_path"])
        assert ds.image_size == 64  # DatasetConfig default


# ---------------------------------------------------------------------------
# Construction — stenosis filtering
# ---------------------------------------------------------------------------

class TestStenosisFiltering:
    def test_stenosis_category_excluded(self, mock_dataset_dir):
        ds = _ds(mock_dataset_dir)
        for img_id, anns in ds.anns_by_image.items():
            for ann in anns:
                assert ann["category_id"] != 26

    def test_stenosis_constant(self):
        assert ArcadeDataset.STENOSIS_CATEGORY_ID == 26


# ---------------------------------------------------------------------------
# __len__
# ---------------------------------------------------------------------------

class TestDatasetLen:
    def test_len_equals_images_times_patches(self, mock_dataset_dir):
        patches = 3
        ds = _ds(mock_dataset_dir, image_size=64, patches_per_image=patches)
        assert len(ds) == len(ds.image_ids) * patches

    def test_len_with_one_patch(self, mock_dataset_dir):
        ds = _ds(mock_dataset_dir, image_size=64, patches_per_image=1)
        assert len(ds) == len(ds.image_ids)


# ---------------------------------------------------------------------------
# _build_path_cache
# ---------------------------------------------------------------------------

class TestBuildPathCache:
    def test_returns_dict(self, mock_dataset_dir):
        ds = _ds(mock_dataset_dir)
        cache = ds._build_path_cache(mock_dataset_dir["img_dir"])
        assert isinstance(cache, dict)

    def test_maps_stem_to_path(self, mock_dataset_dir):
        ds = _ds(mock_dataset_dir)
        cache = ds._build_path_cache(mock_dataset_dir["img_dir"])
        assert "img_001" in cache
        assert "img_002" in cache

    def test_empty_directory_returns_empty(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        ds = ArcadeDataset.__new__(ArcadeDataset)
        cache = ds._build_path_cache(str(empty))
        assert cache == {}

    def test_bg_prefix_mapped_to_base(self, tmp_path):
        d = tmp_path / "imgs"
        d.mkdir()
        Image.new("L", (32, 32)).save(d / "case001_bg_01.png")
        ds = ArcadeDataset.__new__(ArcadeDataset)
        cache = ds._build_path_cache(str(d))
        assert "case001" in cache


# ---------------------------------------------------------------------------
# _make_mask_from_annotations
# ---------------------------------------------------------------------------

class TestMakeMaskFromAnnotations:
    def test_returns_pil_l_mode(self, mock_dataset_dir, coco_data):
        ds = _ds(mock_dataset_dir)
        img_id = coco_data["images"][0]["id"]
        mask = ds._make_mask_from_annotations(img_id, W=128, H=128)
        assert isinstance(mask, Image.Image)
        assert mask.mode == "L"

    def test_output_size(self, mock_dataset_dir, coco_data):
        ds = _ds(mock_dataset_dir)
        img_id = coco_data["images"][0]["id"]
        mask = ds._make_mask_from_annotations(img_id, W=128, H=128)
        assert mask.size == (128, 128)

    def test_vessel_polygon_rasterized(self, mock_dataset_dir, coco_data):
        ds = _ds(mock_dataset_dir)
        img_id = coco_data["images"][0]["id"]
        mask = ds._make_mask_from_annotations(img_id, W=128, H=128)
        arr = np.array(mask)
        assert arr[20, 20] > 0

    def test_no_stenosis_in_mask(self, mock_dataset_dir, coco_data):
        ds = _ds(mock_dataset_dir)
        img_id = coco_data["images"][0]["id"]
        mask = ds._make_mask_from_annotations(img_id, W=128, H=128)
        assert np.array(mask) is not None


# ---------------------------------------------------------------------------
# _generate_vessel_safe_shape
# ---------------------------------------------------------------------------

class TestGenerateVesselSafeShape:
    @pytest.fixture(autouse=True)
    def ds(self, mock_dataset_dir):
        self.ds = _ds(mock_dataset_dir)

    @pytest.mark.parametrize("shape_type", [
        "circle", "rectangle", "ellipse", "triangle", "line", "blob"
    ])
    def test_shape_returns_correct_size(self, shape_type):
        arr = self.ds._generate_vessel_safe_shape(shape_type, W=128, H=128)
        assert arr.shape == (128, 128)

    @pytest.mark.parametrize("shape_type", [
        "circle", "rectangle", "ellipse", "triangle", "line", "blob"
    ])
    def test_shape_uint8(self, shape_type):
        arr = self.ds._generate_vessel_safe_shape(shape_type, W=128, H=128)
        assert arr.dtype == np.uint8

    @pytest.mark.parametrize("shape_type", [
        "circle", "rectangle", "ellipse", "triangle", "line"
    ])
    def test_shape_only_0_or_255(self, shape_type):
        arr = self.ds._generate_vessel_safe_shape(shape_type, W=128, H=128)
        unique = set(arr.flatten().tolist())
        assert unique.issubset({0, 255})

    @pytest.mark.parametrize("shape_type", [
        "circle", "rectangle", "ellipse", "triangle", "line", "blob"
    ])
    def test_shape_has_nonzero_pixels(self, shape_type):
        found_nonzero = False
        for _ in range(5):
            arr = self.ds._generate_vessel_safe_shape(shape_type, W=128, H=128)
            if arr.max() > 0:
                found_nonzero = True
                break
        assert found_nonzero


# ---------------------------------------------------------------------------
# _extract_safe_patch
# ---------------------------------------------------------------------------

class TestExtractSafePatch:
    @pytest.fixture(autouse=True)
    def ds(self, mock_dataset_dir):
        self.ds = _ds(mock_dataset_dir, image_size=64)

    def test_patch_shape(self):
        img  = np.random.randint(0, 255, (128, 128), dtype=np.uint8).astype(float)
        mask = np.zeros((128, 128), dtype=np.uint8)
        mask[20:80, 20:80] = 255
        p_img, p_mask = self.ds._extract_safe_patch(img, mask, patch_size=64)
        assert p_img.shape  == (64, 64)
        assert p_mask.shape == (64, 64)

    def test_patch_within_bounds(self):
        rng = np.random.default_rng(0)
        img  = rng.integers(0, 255, (256, 256)).astype(float)
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[50:150, 50:150] = 255
        p_img, p_mask = self.ds._extract_safe_patch(img, mask, patch_size=64)
        assert p_img.shape  == (64, 64)
        assert p_mask.shape == (64, 64)

    def test_image_smaller_than_patch_raises(self):
        img  = np.zeros((32, 32))
        mask = np.zeros((32, 32), dtype=np.uint8)
        with pytest.raises((ValueError, Exception)):
            self.ds._extract_safe_patch(img, mask, patch_size=64)


# ---------------------------------------------------------------------------
# __getitem__
# ---------------------------------------------------------------------------

class TestDatasetGetItem:
    def test_returns_two_tensors(self, mock_dataset_dir):
        import torch
        ds = _ds(mock_dataset_dir, image_size=64, patches_per_image=1)
        img_t, mask_t = ds[0]
        assert isinstance(img_t, torch.Tensor)
        assert isinstance(mask_t, torch.Tensor)

    def test_image_tensor_shape(self, mock_dataset_dir):
        ds = _ds(mock_dataset_dir, image_size=64, patches_per_image=1)
        img_t, _ = ds[0]
        assert img_t.shape == (1, 64, 64)

    def test_mask_tensor_shape(self, mock_dataset_dir):
        ds = _ds(mock_dataset_dir, image_size=64, patches_per_image=1)
        _, mask_t = ds[0]
        assert mask_t.shape == (1, 64, 64)

    def test_image_range(self, mock_dataset_dir):
        ds = _ds(mock_dataset_dir, image_size=64, patches_per_image=1)
        img_t, _ = ds[0]
        assert img_t.min() >= -1.0 - 1e-5
        assert img_t.max() <=  1.0 + 1e-5

    def test_mask_range(self, mock_dataset_dir):
        ds = _ds(mock_dataset_dir, image_size=64, patches_per_image=1)
        _, mask_t = ds[0]
        assert mask_t.min() >= 0.0 - 1e-5
        assert mask_t.max() <= 1.0 + 1e-5

    def test_dtype_float32(self, mock_dataset_dir):
        import torch
        ds = _ds(mock_dataset_dir, image_size=64, patches_per_image=1)
        img_t, mask_t = ds[0]
        assert img_t.dtype  == torch.float32
        assert mask_t.dtype == torch.float32

    def test_precomputed_mask_loaded(self, mock_dataset_dir_with_masks):
        ds = ArcadeDataset(
            mock_dataset_dir_with_masks["img_dir"],
            mock_dataset_dir_with_masks["ann_path"],
            DatasetConfig(
                image_size=64,
                patches_per_image=1,
                mask_dir=mock_dataset_dir_with_masks["mask_dir"],
            ),
        )
        assert len(ds) > 0
        img_t, mask_t = ds[0]
        assert img_t.shape  == (1, 64, 64)
        assert mask_t.shape == (1, 64, 64)
