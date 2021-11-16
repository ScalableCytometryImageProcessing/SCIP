import dask
import dask.bag
import dask.dataframe
import pandas
import zarr
import numpy
from pathlib import Path


def load_image(event, channels, clip):
    """
    Load an image from a certain path

    Args:
        path (str): path of image,
        z (zarr): zarr object
        channels (list, optional): image channels to load. Defaults to None.

    Returns:
        dict: dictionary containing pixel values (ndarray) and path for each image
    """
    newevent = event.copy()

    i = event["zarr_idx"]
    z = zarr.open(event["path"])
    try:
        arr = z[i].reshape(z.attrs["shape"][i])[channels]
    except ValueError as e:
        print(event)
        raise e
    arr = arr.astype(numpy.float32)

    if clip is not None:
        arr = numpy.clip(arr, 0, clip)

    newevent["pixels"] = arr

    return newevent


def bag_from_directory(path, idx, channels, partition_size, clip):
    """
    Construct delayed ops for all tiffs in a directory

    Args:
        path (str): Directory to find tiffs

    Returns:
        dask.bag: bag containing dictionaries with image data
    """


    def load_image_partition(partition):
        return [load_image(event, channels, clip) for event in partition]

    z = zarr.open(path)
    path = Path(path)
    events = []
    for i, obj in enumerate(z.attrs["object_number"]):
        events.append(dict(
            path=str(path),
            zarr_idx=i,
            idx=f"{idx}_{obj}",
            group=str(path.stem)
        ))

    meta = pandas.DataFrame.from_records(data=events, index="idx")
    meta.columns = [f"meta_{c}" for c in meta.columns]
    meta = dask.dataframe.from_pandas(meta, chunksize=10*partition_size)

    bag = dask.bag.from_sequence(events, partition_size=partition_size)
    bag = bag.map_partitions(load_image_partition)

    return bag, meta, clip
