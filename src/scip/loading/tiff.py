# Copyright (C) 2022 Maxim Lippeveld
#
# This file is part of SCIP.
#
# SCIP is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SCIP is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with SCIP.  If not, see <http://www.gnu.org/licenses/>.

from typing import List, Mapping, Any

import re
import logging
from pathlib import Path
from functools import partial

import pandas
import dask
import dask.bag
import dask.dataframe
import dask.array
import numpy
import tifffile
from aicsimageio.readers.tiff_glob_reader import TiffGlobReader

from scip.segmentation import util
from scip.loading import util as l_util

logging.getLogger("tifffile").setLevel(logging.ERROR)


def _load_image_tiff(event, channels):
    try:
        paths = [event[str(c)] for c in channels]
        arr = tifffile.imread(paths)
        arr = arr.astype(numpy.float32)

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


def _load_block(event, channels, map_to_index):

    paths = [event[str(c)] for c in channels]

    im = TiffGlobReader(
        glob_in=paths,
        indexer=map_to_index
    ).get_image_data("CXY")

    newevent = event.copy()
    newevent["pixels"] = im
    return newevent


def get_loader_meta(**kwargs) -> Mapping[str, type]:
    return dict(path=str)


def _map_to_index(f, regex, channels):
    idx = int(re.search(regex, str(f)).group("channel"))
    m = {c: i for i, c in enumerate(channels)}
    return pandas.Series(dict(S=0, T=0, C=m[idx], Z=0))


@dask.delayed
def _meta_from_directory(regex, path):
    logger = logging.getLogger(__name__)

    path = Path(path)

    matches = []
    i = 0
    for p in path.glob("*.tif*"):
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
    df2 = df.loc[~df.index.duplicated(keep='first'), df.drop(columns=["path", "channel"]).columns]

    df = pandas.concat([df1, df2], axis=1)

    pre_filter = len(df)
    df = df[~df1.isna().any(axis=1)]  # drop rows with missing files
    dropped = pre_filter - len(df)
    logger.warning("Dropped %d rows because of missing channel files in %s" % (dropped, str(path)))

    df["path"] = df.iloc[:, 0]

    return df.to_dict(orient="records")


def bag_from_directory(
    *,
    path: str,
    channels: List[int],
    partition_size: int,
    gpu_accelerated: bool,
    regex: str,
    output: Path,
    segment_method: str,
    segment_kw: Mapping[str, Any],
) -> dask.bag.Bag:

    meta = _meta_from_directory(regex, path)

    bag = dask.bag.from_delayed(meta)
    bag = bag.repartition(npartitions=100)

    if segment_method is not None:

        _m2i = partial(_map_to_index, channels=channels, regex=regex)
        func = partial(_load_block, map_to_index=_m2i)

        bag = bag.map_partitions(l_util._load_image_partition, channels=channels, load=func)

        bag = util.bag_from_blocks(
            blocks=bag,
            gpu_accelerated=gpu_accelerated,
            output=output,
            segment_kw=segment_kw,
            segment_method=segment_method,
            group_keys=[]
        )
    else:
        # bag = dask.bag.from_sequence(records, partition_size=partition_size)
        bag = dask.bag.from_delayed(meta)
        bag = bag.map_partitions(
            l_util._load_image_partition, channels=channels, load=_load_image_tiff)

    return bag
