import pandas
import re
from pathlib import Path
import dask.bag
import tifffile
import logging
logging.getLogger("tifffile").setLevel(logging.ERROR)


def load_image(event, channels): 
    paths = [event[str(c)] for c in channels]
    arr = tifffile.imread(paths) / 2**12
    return dict(pixels=arr, path=paths, idx=event["idx"])


def bag_from_directory(*, path, idx, channels, partition_size, regex):
 
    def match(p):
        m = re.match(regex, str(p)).groupdict()
        if m is not None:
            return {**m, **dict(path=str(p))}
        else:
            return None

    path = Path(path)    
    
    matches = list(filter(lambda r: r is not None, map(match, path.glob("**/*.tif*"))))
    df = pandas.DataFrame.from_dict(matches)
    df = df.pivot(index="id", columns="channel", values="path")
    df["idx"] = pandas.RangeIndex(start=idx, stop=idx+len(df))

    bag = dask.bag.from_sequence(df.to_dict(orient="records"), partition_size=partition_size)

    df = df.set_index("idx")
    df.columns = [f"meta_{c}" for c in df.columns]
    return bag.map_partitions(
        lambda partition: [load_image(event, channels) for event in partition]
    ), df
