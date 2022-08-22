import pytest
from scip.illumination_correction import jones_2006


@pytest.mark.parametrize("fake_images_bag", [False], indirect=True)
def test_correct(fake_images_bag, tmp_path):

    images = jones_2006.correct(
        images=fake_images_bag,
        key="group",
        median_filter_size=100,
        output=tmp_path
    )

    a = images.compute()

    assert len(a) > 0
    assert all("pixels" in im for im in a)
    assert (tmp_path / "correction_images.pickle").exists()
