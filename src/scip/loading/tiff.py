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

"""
Data loader for Tagged Image File format based on aicsimageio package. Expects one channel per file.

Regex pattern for load_pixels and meta_from_directory functions has to contain
channel and id named group.
"""

from typing import Callable, List, Mapping, Any

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
from scip.loading import util as l_util

logging.getLogger("tifffile").setLevel(logging.ERROR)


def _load_image_tiff(
    event: Mapping[str, Any],
    channels: List[int]
):
    """Adds pixels attribute containing pixel values of this event. Values are loaded from
    one file per channel.

    Keyword args:
        event (Mapping[str, Any]): Contains information for an event. Must contain keys
            corresponding to each channel.
        channels (List[int]): List of channels to load.

    Returns:
        event (Mapping[str, Any]): event with pixel values.
    """

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


def _load_block(
    event: Mapping[str, Any],
    channels: List[int],
    map_to_index: Callable
):
    """Adds pixels attribute containing pixel values of this event. Values are loaded from one
    file per channel.

    Keyword args:
        event (Mapping[str, Any]): Contains information for an event. Must contain keys
            corresponding to each channel.
        channels (List[int]): List of channels to load.
        map_to_index (Callable): callable that takes file name as input and outputs pandas Series
            indicating channel index.

    Returns:
        event (Mapping[str, Any]): event with pixel values.
    """

    paths = [event[str(c)] for c in channels]

    im = TiffGlobReader(
        glob_in=paths,
        indexer=map_to_index
    ).get_image_data("CXY")

    newevent = event.copy()
    newevent["pixels"] = im.astype(numpy.float32)
    return newevent


def get_loader_meta(**kwargs) -> Mapping[str, type]:
    """Returns key to type mapping of metadata."""
    return dict(path=str)


def get_group_keys():
    return []


def _map_to_index(f, regex, channels):
    idx = int(re.search(regex, str(f)).group("channel"))
    m = {c: i for i, c in enumerate(channels)}
    return pandas.Series(dict(S=0, T=0, C=m[idx], Z=0))


@dask.delayed
def meta_from_directory(path, regex):
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


def load_pixels(
    images: dask.bag.Bag,
    channels: List[int],
    regex: str,
    **kwargs
) -> dask.bag.Bag:
    _m2i = partial(_map_to_index, channels=channels, regex=regex)
    func = partial(_load_block, map_to_index=_m2i)

    return images.map_partitions(l_util._load_image_partition, channels=channels, load=func)

    # bag = dask.bag.from_delayed(meta)
    # bag = bag.map_partitions(
    #     l_util._load_image_partition, channels=channels, load=_load_image_tiff)
