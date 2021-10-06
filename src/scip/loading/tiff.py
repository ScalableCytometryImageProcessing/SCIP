import pandas
import re
from pathlib import Path
import dask.bag
import dask.dataframe
import tifffile
import logging
logging.getLogger("tifffile").setLevel(logging.ERROR)


def load_image(event, channels):
    try:
        paths = [event[str(c)] for c in channels]
        arr = tifffile.imread(paths) / 2**12
        return dict(pixels=arr, path=paths, idx=event["idx"])
    except TypeError as e:
        logging.getLogger(__name__).exception(e)
        logging.getLogger(__name__).error(paths)
        raise e


def bag_from_directory(*, path, idx, channels, partition_size, regex):

    logger = logging.getLogger(__name__)

    def match(p):
        m = re.match(regex, str(p)).groupdict()
        if m is not None:
            return {**m, **dict(path=str(p))}
        else:
            return None

    def load_image_partition(partition):
        return [load_image(event, channels) for event in partition]

    path = Path(path)

    matches = list(filter(lambda r: r is not None, map(match, path.glob("*.tif*"))))
    df = pandas.DataFrame.from_dict(matches)

    df = df.pivot(index="id", columns="channel", values="path")

    pre_filter = len(df)
    df = df.dropna(axis=0, how="any")
    dropped = pre_filter - len(df)
    logger.warning("Dropped %d rows because of missing channel files in %s." % (dropped, str(path)))

    df["idx"] = pandas.RangeIndex(start=idx, stop=idx + len(df))

    bag = dask.bag.from_sequence(df.to_dict(orient="records"), partition_size=partition_size)
    bag = bag.map_partitions(load_image_partition)

    df = df.set_index("idx")
    df.columns = [f"meta_{c}" for c in df.columns]
    meta = dask.dataframe.from_pandas(df, chunksize=partition_size)
    return bag, meta
