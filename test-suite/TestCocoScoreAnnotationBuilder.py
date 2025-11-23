import pytest

from utils.CocoBuilder import CocoScoreAnnotationBuilder


def test_build_requires_id_and_image_id():
    b = CocoScoreAnnotationBuilder()
    # neither id nor image_id set -> should raise
    with pytest.raises(ValueError):
        b.build()

    # set only id -> still missing image_id
    b.with_id(1)
    with pytest.raises(ValueError):
        b.build()

    # set only image_id -> still missing id
    b2 = CocoScoreAnnotationBuilder()
    b2.with_image_id(10)
    with pytest.raises(ValueError):
        b2.build()


def test_build_sets_all_fields_correctly():
    b = CocoScoreAnnotationBuilder()
    b.with_id(5)
    b.with_image_id(7)
    b.with_score(0.42)
    b.with_initial_score(0.5)
    b.with_category_id(None)  # should set category_id to 0
    b.with_sequence(2)
    b.with_transformation("TF_TEST")

    ann = b.build()

    assert ann["id"] == 5
    assert ann["image_id"] == 7
    assert ann["score"] == 0.42
    assert ann["initial_score"] == 0.5
    assert ann["category_id"] == 0
    assert ann["sequence"] == 2
    assert ann["transformation"] == "TF_TEST"

