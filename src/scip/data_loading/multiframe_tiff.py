import dask
from PIL import Image
import dask.bag
import numpy
from pathlib import Path


def load_image(path, channels=None):
    """
    Load an image from a certain path

    Args:
        path (str): path of image
        channels (list, optional): image channels to load. Defaults to None.

    Returns:
        dict: dictionary containing pixel values (ndarray) and path for each image
    """
    im = Image.open(path)

    if channels is None:
        channels = range(im.n_frames)

    arr = numpy.empty(shape=(len(channels), im.height, im.width), dtype=float)
    for i in channels:
        im.seek(i)
        arr[i] = numpy.array(im)
    return dict(pixels=arr, path=path)


def bag_from_directory(path, channels, partition_size):
    """
    Construct delayed ops for all tiffs in a directory

    Args:
        path (str): Directory to find tiffs

    Returns:
        dask.bag: bag containing dictionaries with image data
    """

    image_paths = []
    for p in Path(path).glob("**/*.tiff"):
        image_paths.append(str(p))

    bag = dask.bag.from_sequence(image_paths, partition_size=partition_size)
    return bag.map_partitions(
        lambda paths: [load_image(path, channels) for path in paths]
    )
