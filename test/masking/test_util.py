from scip.masking import util
import pytest


@pytest.mark.parametrize("fake_images_bag", [True], indirect=True)
def test_apply(fake_images_bag):

    bag = fake_images_bag.map(util.apply)
    bag = bag.compute()

    assert len(bag) > 0

    b = bag[0]
    assert all([k in b for k in ["combined_background", "background", "mask", "combined_mask"]])
    assert len(b["mask"].shape) == 3
    assert len(b["combined_mask"].shape) == 2
    assert len(b["combined_background"]) == len(b["background"])
