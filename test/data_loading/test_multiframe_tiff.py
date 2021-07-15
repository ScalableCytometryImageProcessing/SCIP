from pathlib import Path
from sip.data_loading import multiframe_tiff


def test_load_image(data: Path):
    path = data / "1.tiff"
    im = multiframe_tiff.load_image(str(path))

    assert "path" in im
    assert "pixels" in im
    assert im["path"] == str(path)
    assert im["pixels"].shape == (8, 29, 104)
    assert im["pixels"].mean() > 0


def test_bag_from_directory(data, cluster):
    bag = multiframe_tiff.bag_from_directory(data, partition_size=2)
    images = bag.compute()
    assert len(images) == 2
