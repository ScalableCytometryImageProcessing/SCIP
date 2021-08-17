from pathlib import Path
from scip.data_loading import multiframe_tiff


def test_load_image(data: Path):
    event = dict(path=data / "1.tiff", idx=0)
    im = multiframe_tiff.load_image(event)

    assert "path" in im
    assert "pixels" in im
    assert str(im["path"]) == str(event["path"])
    assert im["pixels"].shape == (8, 29, 104)
    assert im["pixels"].mean() > 0


def test_bag_from_directory(data, cluster):
    bag, num = multiframe_tiff.bag_from_directory(data, idx=0, channels=None, partition_size=2)
    images = bag.compute()
    assert len(images) == 2
    assert len(images) == num
    assert all(len(im["pixels"]) == 8 for im in images)
