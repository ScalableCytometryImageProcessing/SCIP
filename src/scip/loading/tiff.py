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

        if clip is not None:
            arr = numpy.clip(arr, 0, clip)

        # tifffile collapses axis with size 1,
        # occurrs when only one path is passed
        if len(arr.shape) < 3:
            arr = arr[numpy.newaxis, ...]

        newevent = event.copy()
        newevent["pixels"] = arr.astype(float)
        return newevent
    except TypeError as e:
        logging.getLogger(__name__).exception(e)
        logging.getLogger(__name__).error(paths)
        raise e


def bag_from_directory(*, path, idx, channels, partition_size, regex, clip):

    logger = logging.getLogger(__name__)

    def match(p):
        m = re.search(regex, str(p))
        if m is not None:
            groups = m.groupdict()
            gid = groups["id"]
            groups["id"] = f"{idx}_{gid}"
            return {**groups, **dict(path=str(p))}
        else:
            return None

    def load_image_partition(partition):
        return [load_image(event, channels, clip) for event in partition]

    path = Path(path)

    matches = list(filter(lambda r: r is not None, map(match, path.glob("*.tif*"))))
    df = pandas.DataFrame.from_dict(matches)
    df1 = df.pivot(index="id", columns="channel", values="path")
    df = df.set_index("id")
    df2 = df.loc[~df.index.duplicated(keep='first'), df.drop(columns=["path"]).columns]

    df = pandas.concat([df1, df2], axis=1)
    df.index.name = "idx"

    pre_filter = len(df)
    df = df[~df1.isna().any(axis=1)]
    dropped = pre_filter - len(df)
    logger.warning("Dropped %d rows because of missing channel files in %s" % (dropped, str(path)))

    bag = dask.bag.from_sequence(
        df.reset_index(drop=False).to_dict(orient="records"), partition_size=partition_size)
    bag = bag.map_partitions(load_image_partition)

    df.columns = [f"meta_{c}" for c in df.columns]
    meta = dask.dataframe.from_pandas(df, chunksize=partition_size)
    return bag, meta
