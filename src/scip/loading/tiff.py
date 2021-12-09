import pandas
import re
from pathlib import Path
import dask.bag
import dask.dataframe
import tifffile
import logging
import numpy
logging.getLogger("tifffile").setLevel(logging.ERROR)


def load_image(event, channels, clip):
    try:
        paths = [event[str(c)] for c in channels]
        arr = tifffile.imread(paths)
        arr = arr.astype(numpy.float32)

        if clip is not None:
            arr = numpy.clip(arr, 0, clip)

        # tifffile collapses axis with size 1,
        # occurrs when only one path is passed
        if len(arr.shape) < 3:
            arr = arr[numpy.newaxis, ...]

        newevent = event.copy()
        newevent["pixels"] = arr
        return newevent
    except TypeError as e:
        logging.getLogger(__name__).exception(e)
        logging.getLogger(__name__).error(paths)
        raise e


def bag_from_directory(*, path, channels, partition_size, gpu_accelerated, regex, clip):

    logger = logging.getLogger(__name__)

    def load_image_partition(partition):
        return [load_image(event, channels, clip) for event in partition]

    path = Path(path)

    matches = []
    i = 0
    for p in path.glob("*.tif"):
        m = re.search(regex, str(p))
        if m is not None:
            groups = m.groupdict()
            matches.append({
                **groups,
                **dict(path=str(p))
            })
            i += 1

    df = pandas.DataFrame.from_dict(matches)
    df1 = df.pivot(index="id", columns="channel", values="path")
    df = df.set_index("id")
    df2 = df.loc[~df.index.duplicated(keep='first'), df.drop(columns=["path"]).columns]

    df = pandas.concat([df1, df2], axis=1)

    pre_filter = len(df)
    df = df[~df1.isna().any(axis=1)]
    dropped = pre_filter - len(df)
    logger.warning("Dropped %d rows because of missing channel files in %s" % (dropped, str(path)))

    bag = dask.bag.from_sequence(
        df.to_dict(orient="records"), partition_size=partition_size)
    bag = bag.map_partitions(load_image_partition)

    loader_meta = {c: str for c in df.columns}
    return bag, loader_meta, clip
