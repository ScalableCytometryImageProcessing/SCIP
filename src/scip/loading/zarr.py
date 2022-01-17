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

import dask
import dask.bag
import dask.dataframe
import zarr
import numpy
from pathlib import Path
import re
from typing import Tuple
import copy


def reload_image_partition(
    partition,
    channels,
    clip: int,
    regex: str,
    limit: int = -1
):
    z = zarr.open(partition[0]["path"])
    indices = [p["zarr_idx"] for p in partition]
    data = z.get_coordinate_selection(indices)
    shapes = numpy.array(z.attrs["shape"])[indices]

    newpartition = copy.deepcopy(partition)
    for i in range(len(partition)):
        if "mask" not in partition[i]:
            continue

        if clip is not None:
            newpartition[i]["pixels"] = numpy.clip(data[i].reshape(shapes[i])[channels], 0, clip)
        else:
            newpartition[i]["pixels"] = data[i].reshape(shapes[i])[channels]
        newpartition[i]["pixels"] = newpartition[i]["pixels"].astype(numpy.float32)
    return newpartition


def load_image_partition(partition, z, channels, clip):

    start, end = partition[0]["zarr_idx"], partition[-1]["zarr_idx"]
    data = z[start:end + 1]
    shapes = z.attrs["shape"][start:end + 1]

    for i in range(len(partition)):
        if clip is not None:
            partition[i]["pixels"] = numpy.clip(data[i].reshape(shapes[i])[channels], 0, clip)
        else:
            partition[i]["pixels"] = data[i].reshape(shapes[i])[channels]
        partition[i]["pixels"] = partition[i]["pixels"].astype(numpy.float32)
    return partition


def bag_from_directory(
    *,
    path: str,
    channels: list,
    partition_size: int,
    gpu_accelerated: bool,
    limit: int = -1,
    clip: int,
    regex: str,
) -> Tuple[dask.bag.Bag, dask.dataframe.DataFrame, int, int]:

    """
    Construct delayed ops for all tiffs in a directory

    Args:
        path (str): Directory to find tiffs

    Returns:
        dask.bag: bag containing dictionaries with image data
    """

    match = re.search(regex, str(path))
    groups = match.groupdict()

    z = zarr.open(path)
    path = Path(path)
    events = []

    if limit == -1:
        limit = len(z)

    for i, obj in enumerate(z.attrs["object_number"][:limit]):
        events.append({**groups, **{
            "path": str(path),
            "zarr_idx": i,
            "object_number": obj
        }})

    bag = dask.bag.from_sequence(events, partition_size=partition_size)
    bag = bag.map_partitions(load_image_partition, z, channels, clip)

    loader_meta = dict(path=str, zarr_idx=int, object_number=int)
    for k in groups.keys():
        loader_meta[k] = str
    return bag, loader_meta, clip, len(events)
