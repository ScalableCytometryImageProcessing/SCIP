from pathlib import Path
from scip.loading import multiframe_tiff


def test_load_image(image_path: Path):
    event = dict(path=image_path)
    im = multiframe_tiff.load_image(event)

    assert "path" in im
    assert "pixels" in im
    assert str(im["path"]) == str(event["path"])
    assert im["pixels"].mean() > 0


def test_bag_from_directory(images_folder, cluster):
    bag, meta = multiframe_tiff.bag_from_directory(
        images_folder, channels=None, partition_size=2)
    images = bag.compute()
    assert len(images) == 11
    assert len(images) == len(meta)
    assert all(len(im["pixels"]) == 8 for im in images)
