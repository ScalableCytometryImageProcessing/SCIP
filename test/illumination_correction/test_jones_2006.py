import pytest
import numpy
import pickle
from scip.illumination_correction import jones_2006


@pytest.mark.parametrize("fake_images_bag", [False], indirect=True)
def test_correct(fake_images_bag, tmp_path):

    images = jones_2006.correct(
        images=fake_images_bag,
        key="group",
        median_filter_size=5,
        output=tmp_path
    )

    original_images = fake_images_bag.compute()
    a = images.compute()

    assert len(a) > 0
    assert all("pixels" in im for im in a)
    assert all(~numpy.isnan(im["pixels"]).any() for im in a)

    # check if images are changed after correction
    assert all(
        numpy.sum(im1["pixels"]) != numpy.sum(im2["pixels"])
        for im1, im2 in zip(original_images, a)
    )

    assert (tmp_path / "correction_images.pickle").exists()

    with open(tmp_path / "correction_images.pickle", "rb") as fh:
        corr = pickle.load(fh)
    assert all(~(v==1).all() for _, v in corr.items())
