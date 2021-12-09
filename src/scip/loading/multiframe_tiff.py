import dask
from PIL import Image
import dask.bag
import dask.dataframe
import numpy
from pathlib import Path
import pandas


def load_image(event, channels=None):
    """
    Load an image from a certain path

    Args:
        path (str): path of image
        channels (list, optional): image channels to load. Defaults to None.

    Returns:
        dict: dictionary containing pixel values (ndarray) and path for each image
    """
    im = Image.open(event["path"])

    if channels is None:
        channels = range(im.n_frames)

    arr = numpy.empty(shape=(len(channels), im.height, im.width), dtype=float)
    for i, c in enumerate(channels):
        im.seek(c)
        arr[i] = numpy.array(im)

    newevent = event.copy()
    newevent["pixels"] = arr
    return newevent


def bag_from_directory(path, channels, partition_size):
    """
    Construct delayed ops for all tiffs in a directory

    Args:
        path (str): Directory to find tiffs

    Returns:
        dask.bag: bag containing dictionaries with image data
    """

    def load_image_partition(partition):
        return [load_image(event, channels) for event in partition]

    events = []
    for i, p in enumerate(Path(path).glob("**/*.tiff")):
        events.append(dict(path=str(p), group=str(p.parent)))

    meta = pandas.DataFrame.from_records(data=events)
    meta.columns = [f"meta_{c}" for c in meta.columns]
    meta = dask.dataframe.from_pandas(meta, chunksize=partition_size)

    bag = dask.bag.from_sequence(events, partition_size=partition_size)
    bag = bag.map_partitions(load_image_partition)

    return bag, meta
