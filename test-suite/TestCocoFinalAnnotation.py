from pathlib import Path

from utils.CocoBuilder import CocoBuilder


def test_add_image_final_score_annotation_sets_fields_correctly():
    cb = CocoBuilder("final_test")
    image_id = cb.add_image("img_final.png", 10, 10)

    # initial score provided
    ann_id = cb.add_image_final_score_annotation(image_id, score=0.75, initial_score=0.5)
    assert len(cb.annotations) == 1
    ann = cb.annotations[0]

    # category_id should be 0 (image-level)
    assert ann.get("category_id") == 0
    # sequence should not be present
    assert "sequence" not in ann
    # score and initial_score should be present and match
    assert ann.get("score") == 0.75
    assert ann.get("initial_score") == 0.5


def test_add_image_final_score_annotation_without_initial():
    cb = CocoBuilder("final_test2")
    image_id = cb.add_image("img_final2.png", 20, 20)

    ann_id = cb.add_image_final_score_annotation(image_id, score=0.33)
    assert len(cb.annotations) == 1
    ann = cb.annotations[0]

    assert ann.get("category_id") == 0
    assert "sequence" not in ann
    assert ann.get("score") == 0.33
    # initial_score should not be present
    assert "initial_score" not in ann

