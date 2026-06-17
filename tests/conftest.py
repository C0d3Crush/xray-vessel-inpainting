import sys
import os
import json
import pytest
import numpy as np
from pathlib import Path
from PIL import Image

# Make src/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# ---------------------------------------------------------------------------
# COCO annotation helpers
# ---------------------------------------------------------------------------

COCO_DATA = {
    "images": [
        {"id": 1, "file_name": "img_001.png", "width": 128, "height": 128},
        {"id": 2, "file_name": "img_002.png", "width": 128, "height": 128},
    ],
    "annotations": [
        {
            "id": 1, "image_id": 1, "category_id": 1,
            "segmentation": [[10.0, 10.0, 50.0, 10.0, 50.0, 50.0, 10.0, 50.0]],
        },
        {
            "id": 2, "image_id": 2, "category_id": 1,
            "segmentation": [[60.0, 60.0, 100.0, 60.0, 100.0, 100.0, 60.0, 100.0]],
        },
        # Stenosis annotation (category 26) — must be filtered out
        {
            "id": 3, "image_id": 1, "category_id": 26,
            "segmentation": [[5.0, 5.0, 15.0, 5.0, 15.0, 15.0, 5.0, 15.0]],
        },
    ],
    "categories": [
        {"id": 1, "name": "vessel"},
        {"id": 26, "name": "stenosis"},
    ],
}


@pytest.fixture(scope="session")
def coco_data():
    return COCO_DATA


@pytest.fixture
def mock_dataset_dir(tmp_path, coco_data):
    """Temp directory with two 128×128 grayscale PNGs and a COCO JSON."""
    img_dir = tmp_path / "images"
    ann_dir = tmp_path / "annotations"
    img_dir.mkdir()
    ann_dir.mkdir()

    rng = np.random.default_rng(0)
    for img_info in coco_data["images"]:
        arr = rng.integers(50, 200, (128, 128), dtype=np.uint8)
        Image.fromarray(arr, mode="L").save(img_dir / img_info["file_name"])

    ann_path = ann_dir / "annotations.json"
    ann_path.write_text(json.dumps(coco_data))

    return {"img_dir": str(img_dir), "ann_path": str(ann_path)}


@pytest.fixture
def mock_dataset_dir_with_masks(mock_dataset_dir, coco_data, tmp_path):
    """Extends mock_dataset_dir with a precomputed mask directory."""
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()

    rng = np.random.default_rng(1)
    for img_info in coco_data["images"]:
        arr = rng.integers(0, 2, (128, 128), dtype=np.uint8) * 255
        Image.fromarray(arr, mode="L").save(mask_dir / img_info["file_name"])

    return {**mock_dataset_dir, "mask_dir": str(mask_dir)}
