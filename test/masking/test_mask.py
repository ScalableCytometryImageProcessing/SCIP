from scip.masking import sobel
import pytest


@pytest.mark.parametrize("fake_images_bag", [(False, True)], indirect=True)
def test_sobel(fake_images_bag):
    images = sobel.create_masks_on_bag(bag=fake_images_bag, main_channel=0)

    images = images.compute()

    assert len(images) > 0
    assert all("mask" in im for im in images)
    assert all(im["mask"].any() for im in images)
    assert all(im["mask"].shape == im["pixels"].shape for im in images)
