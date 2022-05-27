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

from typing import List

import re
import logging
from pathlib import Path

import pandas
import dask.bag
import dask.dataframe
import tifffile
import numpy

logging.getLogger("tifffile").setLevel(logging.ERROR)


def _load_image(event, channels):
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


def _load_image_partition(partition, channels):
    return [_load_image(event, channels) for event in partition]


def bag_from_directory(
    *,
    path: str,
    output: Path,
    channels: List[int],
    partition_size: int,
    gpu_accelerated: bool,
    limit: int = -1,
    regex: str
):

    logger = logging.getLogger(__name__)

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

    if limit != -1:
        df = df.iloc[:limit]

    bag = dask.bag.from_sequence(
        df.to_dict(orient="records"), partition_size=partition_size)
    bag = bag.map_partitions(_load_image_partition, channels=channels)

    loader_meta = {c: str for c in df.columns}
    return bag, loader_meta, len(df)
