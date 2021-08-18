from pathlib import Path
from scip.data_loading import multiframe_tiff


def test_load_image(image_path: Path):
    event = dict(path=image_path, idx=0)
    im = multiframe_tiff.load_image(event)

    assert "path" in im
    assert "pixels" in im
    assert str(im["path"]) == str(event["path"])
    assert im["pixels"].mean() > 0


def test_bag_from_directory(images_folder, cluster):
    bag, num = multiframe_tiff.bag_from_directory(
        images_folder, idx=0, channels=None, partition_size=2)
    images = bag.compute()
    assert len(images) == 11
    assert len(images) == num
    assert all(len(im["pixels"]) == 8 for im in images)
