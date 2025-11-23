import json
from pathlib import Path

from utils.CocoBuilder import CocoBuilder
from dataset.Utils import TRANSFORMER_CATEGORY_NAME


def test_sequence_and_transformation_field(tmp_path: Path):
    cb = CocoBuilder("test_dataset")

    # add image
    image_id = cb.add_image("img1.png", 100, 200)

    # add two transformation score annotations with transformer names
    ann1_id = cb.add_image_transformation_score_annotation(image_id, score=0.5, initial_score=0.6, transformer_name="TF_A")
    ann2_id = cb.add_image_transformation_score_annotation(image_id, score=0.7, initial_score=0.5, transformer_name="TF_B")

    # There should be two annotations
    assert len(cb.annotations) == 2

    ann1 = cb.annotations[0]
    ann2 = cb.annotations[1]

    # sequence should increment per image
    assert ann1.get("sequence") == 1
    assert ann2.get("sequence") == 2

    # transformation field should be present and correct
    assert ann1.get("transformation") == "TF_A"
    assert ann2.get("transformation") == "TF_B"

    # categories should contain transformer supercategory and the two transformer categories
    cat_names = {c["name"] for c in cb.categories}
    assert TRANSFORMER_CATEGORY_NAME in cat_names
    assert "TF_A" in cat_names
    assert "TF_B" in cat_names

    # category IDs in annotations should point to existing categories
    cat_ids = {c["id"] for c in cb.categories}
    assert ann1.get("category_id") in cat_ids
    assert ann2.get("category_id") in cat_ids


def test_no_transformer_name_sets_no_transformation(tmp_path: Path):
    cb = CocoBuilder("test_dataset2")
    image_id = cb.add_image("img2.png", 50, 50)

    ann_id = cb.add_image_transformation_score_annotation(image_id, score=0.3, initial_score=0.4)
    assert len(cb.annotations) == 1
    ann = cb.annotations[0]

    # sequence should be 1
    assert ann.get("sequence") == 1

    # transformation key should not be present
    assert "transformation" not in ann

    # category_id should default to 0 (builder uses 0 for None)
    assert ann.get("category_id") == 0

