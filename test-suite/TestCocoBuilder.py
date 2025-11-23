import json
from pathlib import Path

from dataset.Utils import TRANSFORMER_CATEGORY_NAME
from utils.CocoBuilder import CocoBuilder


def test_sequence_and_transformation_field(tmp_path: Path):
    cb = CocoBuilder("test_dataset")

    # add image
    image_id = cb.add_image("img1.png", 100, 200)

    # add two transformation score annotations with transformer names
    ann1_id = cb.add_image_transformation_score_annotation(image_id, score=0.5, initial_score=0.6,
                                                           transformer_name="TF_A")
    ann2_id = cb.add_image_transformation_score_annotation(image_id, score=0.7, initial_score=0.5,
                                                           transformer_name="TF_B")

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


def test_add_image_id_and_map_behavior():
    cb = CocoBuilder("dset")
    id1 = cb.add_image("a.png", 10, 10)
    id2 = cb.add_image("b.png", 20, 20)
    # adding same filename returns same id
    id1_again = cb.add_image("a.png", 10, 10)
    assert id1 == id1_again
    assert id1 != id2
    # internal mapping contains filenames
    assert "a.png" in cb._image_id_map
    assert cb._image_id_map["b.png"] == id2


def test_categories_and_explicit_id_assignment():
    cb = CocoBuilder("dset2")
    # add category without id
    cat1 = cb.add_category("catA")
    # add category with explicit id
    cat2 = cb.add_category("catB", category_id := 42)
    # ensure both present
    names = {c['name'] for c in cb.categories}
    ids = {c['id'] for c in cb.categories}
    assert "catA" in names
    assert "catB" in names
    assert category_id in ids


def test_save_creates_valid_coco_json(tmp_path: Path):
    cb = CocoBuilder("myds")
    cb.set_description("desc")
    img_id = cb.add_image("img.png", 5, 5)
    cb.add_image_transformation_annotation(img_id, "TF_X")
    cb.add_image_transformation_score_annotation(img_id, score=0.1, initial_score=0.2, transformer_name="TF_X")

    target = tmp_path / "annotations.json"
    cb.save(str(target))
    assert target.exists()
    content = json.loads(target.read_text(encoding='utf-8'))
    assert "images" in content and isinstance(content["images"], list)
    assert "annotations" in content and isinstance(content["annotations"], list)
    assert "categories" in content and isinstance(content["categories"], list)
    assert len(content["images"]) == 1
    assert len(content["annotations"]) >= 1


def test_sequence_counter_is_per_image():
    cb = CocoBuilder("dset-seq")
    id_a = cb.add_image("a.png", 10, 10)
    id_b = cb.add_image("b.png", 10, 10)

    # add annotations interleaved
    cb.add_image_transformation_annotation(id_a, "T1")
    cb.add_image_transformation_annotation(id_b, "T2")
    cb.add_image_transformation_annotation(id_a, "T3")

    # inspect sequences
    seqs_a = [ann['sequence'] for ann in cb.annotations if ann['image_id'] == id_a]
    seqs_b = [ann['sequence'] for ann in cb.annotations if ann['image_id'] == id_b]
    assert seqs_a == [1, 2]
    assert seqs_b == [1]
