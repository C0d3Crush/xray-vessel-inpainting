"""Shared COCO annotation loading utilities for scripts."""
import json
from collections import defaultdict

STENOSIS_CATEGORY_ID = 26


def load_coco_annotations(ann_path):
    """
    Parse a COCO JSON file and return annotation lookups.

    Stenosis annotations (category 26) are excluded.

    Returns:
        id_to_info (dict): image_id -> image info dict
        anns_by_image (defaultdict): image_id -> list of annotations
        image_ids (list): image IDs that have at least one annotation
    """
    with open(ann_path) as f:
        coco = json.load(f)

    id_to_info = {img['id']: img for img in coco['images']}
    anns_by_image = defaultdict(list)

    for ann in coco['annotations']:
        if ann['category_id'] != STENOSIS_CATEGORY_ID:
            anns_by_image[ann['image_id']].append(ann)

    image_ids = [img_id for img_id in id_to_info if anns_by_image[img_id]]
    return id_to_info, anns_by_image, image_ids
