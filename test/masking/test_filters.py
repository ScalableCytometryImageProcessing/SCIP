import pytest
from scip.masking import compute_filters


@pytest.mark.parametrize(
    "fake_images_bag, method, settings",
    [(False, "normaltest", {}), (False, "std", dict(threshold=2))],
    indirect=["fake_images_bag"]
)
def test_filters(fake_images_bag, method, settings):
    images = fake_images_bag.map_partitions(
        compute_filters,
        config=[dict(
            method=method,
            settings=settings,
            channel_indices=[0]
        )],
        main_channel_index=0
    )
    images = images.compute()

    assert len(images) > 0
    assert all("mask_filter" in im for im in images)
    assert all(len(im["mask_filter"]) == len(im["pixels"]) for im in images)
