from scip.loading import tiff
import dask.bag
import pytest


@pytest.mark.parametrize("channels, expected_length", [([1], 1), ([1, 2], 2)])
def test_load_pixels(tiffs_folder, channels, expected_length):
    images = tiff.meta_from_directory(
        path=tiffs_folder,
        regex="^.+/test(?P<id>.+)_(?P<channel>[0-9]).+$"
    )
    images = dask.bag.from_delayed(images)
    images = tiff.load_pixels(
        images=images,
        channels=channels,
        regex="^.+/test(?P<id>.+)_(?P<channel>[0-9]).+$"
    )
    images = images.compute()

    assert len(images) > 0
    assert all(len(im["pixels"]) == expected_length for im in images)
