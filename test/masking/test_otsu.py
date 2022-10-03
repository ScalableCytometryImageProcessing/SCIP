from scip.masking import otsu
import pytest
import numpy


@pytest.mark.parametrize("fake_images_bag", [False], indirect=True)
def test_otsu(fake_images_bag):

    def t(a):
        a["mask_filter"] = [True] * len(a["pixels"])
        a["pixels"] = a["pixels"].astype(numpy.uint8)
        return a

    images = fake_images_bag.map(t)
    images = otsu.create_masks_on_bag(images)

    images = images.compute()

    assert len(images) > 0
    assert all("mask" in im for im in images)
