import sys
import os
import json
import pickle
import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for CI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts', 'legacy'))


# ---------------------------------------------------------------------------
# coco_utils
# ---------------------------------------------------------------------------

class TestCocoUtils:
    @pytest.fixture
    def coco_json(self, tmp_path):
        data = {
            "images": [
                {"id": 1, "file_name": "a.png", "width": 128, "height": 128},
                {"id": 2, "file_name": "b.png", "width": 128, "height": 128},
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1,
                 "segmentation": [[10, 10, 50, 10, 50, 50, 10, 50]]},
                {"id": 2, "image_id": 2, "category_id": 26,  # stenosis — must be filtered
                 "segmentation": [[0, 0, 10, 0, 10, 10]]},
            ],
        }
        p = tmp_path / "ann.json"
        p.write_text(json.dumps(data))
        return str(p)

    def test_returns_three_values(self, coco_json):
        from coco_utils import load_coco_annotations
        result = load_coco_annotations(coco_json)
        assert len(result) == 3

    def test_id_to_info_contains_all_images(self, coco_json):
        from coco_utils import load_coco_annotations
        id_to_info, _, _ = load_coco_annotations(coco_json)
        assert 1 in id_to_info and 2 in id_to_info

    def test_stenosis_filtered_out(self, coco_json):
        from coco_utils import load_coco_annotations
        _, anns_by_image, image_ids = load_coco_annotations(coco_json)
        # Image 2 only had stenosis annotation → no vessel annotation → not in image_ids
        assert 2 not in image_ids
        # Image 1 has vessel annotation → in image_ids
        assert 1 in image_ids

    def test_image_ids_only_annotated(self, coco_json):
        from coco_utils import load_coco_annotations
        _, _, image_ids = load_coco_annotations(coco_json)
        assert isinstance(image_ids, list)
        assert len(image_ids) == 1


# ---------------------------------------------------------------------------
# cache_masks — rasterize_mask
# ---------------------------------------------------------------------------

class TestCacheMasks:
    def test_rasterize_mask_shape(self):
        from cache_masks import rasterize_mask
        anns = [{"segmentation": [[0, 0, 64, 0, 64, 64, 0, 64]]}]
        mask = rasterize_mask(anns, 128, 128)
        assert mask.shape == (128, 128)
        assert mask.dtype == np.uint8

    def test_rasterize_mask_filled_inside(self):
        from cache_masks import rasterize_mask
        anns = [{"segmentation": [[10, 10, 50, 10, 50, 50, 10, 50]]}]
        mask = rasterize_mask(anns, 100, 100)
        assert mask[30, 30] == 255   # inside polygon

    def test_rasterize_mask_empty_outside(self):
        from cache_masks import rasterize_mask
        anns = [{"segmentation": [[10, 10, 50, 10, 50, 50, 10, 50]]}]
        mask = rasterize_mask(anns, 100, 100)
        assert mask[0, 0] == 0       # outside polygon

    def test_rasterize_mask_empty_annotations(self):
        from cache_masks import rasterize_mask
        mask = rasterize_mask([], 64, 64)
        assert mask.sum() == 0

    def test_rasterize_mask_short_polygon_skipped(self):
        from cache_masks import rasterize_mask
        # polygon with < 3 points must be skipped gracefully
        anns = [{"segmentation": [[0, 0, 10, 10]]}]
        mask = rasterize_mask(anns, 64, 64)
        assert mask.sum() == 0


# ---------------------------------------------------------------------------
# preprocess_coco
# ---------------------------------------------------------------------------

class TestPreprocessCoco:
    @pytest.fixture
    def coco_json(self, tmp_path):
        data = {
            "images": [{"id": 1, "file_name": "x.png", "width": 64, "height": 64}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1,
                 "segmentation": [[0, 0, 32, 0, 32, 32, 0, 32]]},
            ],
            "categories": [{"id": 1, "name": "vessel"}],
        }
        p = tmp_path / "ann.json"
        p.write_text(json.dumps(data))
        return str(p)

    def test_returns_expected_keys(self, coco_json):
        from preprocess_coco import preprocess_coco
        result = preprocess_coco(coco_json)
        assert set(result.keys()) >= {'id_to_info', 'anns_by_image', 'image_ids', 'categories'}

    def test_image_ids_match(self, coco_json):
        from preprocess_coco import preprocess_coco
        result = preprocess_coco(coco_json)
        assert 1 in result['image_ids']

    def test_pickle_roundtrip(self, tmp_path, coco_json):
        from preprocess_coco import preprocess_coco
        result = preprocess_coco(coco_json)
        pkl_path = tmp_path / "out.pkl"
        with open(pkl_path, 'wb') as f:
            pickle.dump(result, f)
        with open(pkl_path, 'rb') as f:
            loaded = pickle.load(f)
        assert loaded['image_ids'] == result['image_ids']


# ---------------------------------------------------------------------------
# plot_training
# ---------------------------------------------------------------------------

class TestPlotTraining:
    @pytest.fixture
    def legacy_csv(self, tmp_path):
        p = tmp_path / "log.csv"
        p.write_text("epoch,train_loss,val_psnr,val_ssim\n1,0.5,30.0,0.85\n2,0.4,31.0,0.87\n")
        return str(p)

    @pytest.fixture
    def extended_csv(self, tmp_path):
        p = tmp_path / "log.csv"
        p.write_text(
            "epoch,train_loss,val_psnr,val_ssim,val_wasserstein,val_rmse\n"
            "1,0.5,30.0,0.85,1.2,5.0\n"
            "2,0.4,31.0,0.87,1.1,4.8\n"
        )
        return str(p)

    def test_legacy_csv_saves_file(self, tmp_path, legacy_csv):
        from plot_training import plot_training_log
        out = str(tmp_path / "plot.png")
        plot_training_log(legacy_csv, out)
        assert os.path.exists(out)

    def test_extended_csv_saves_file(self, tmp_path, extended_csv):
        from plot_training import plot_training_log
        out = str(tmp_path / "plot.png")
        plot_training_log(extended_csv, out)
        assert os.path.exists(out)
