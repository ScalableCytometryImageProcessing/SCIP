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

from typing import Mapping, List, Any

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
    partition: List,
    channels: List[int],
    regex: str,
    limit: int = -1
):
    z = zarr.open(partition[0]["path"])
    indices = [p["zarr_idx"] for p in partition]
    data = z.get_coordinate_selection(indices)
    shapes = numpy.array(z.attrs["shape"])[indices]

    newpartition = copy.deepcopy(partition)
    for i in range(len(partition)):
        if "mask" in partition[i]:
            newpartition[i]["pixels"] = data[i].reshape(shapes[i])[channels].astype(numpy.float32)
    return newpartition


def load_image_partition(partition, z, channels):

    start, end = partition[0]["zarr_idx"], partition[-1]["zarr_idx"]
    data = z[start:end + 1]
    shapes = z.attrs["shape"][start:end + 1]

    for i in range(len(partition)):
        partition[i]["pixels"] = data[i].reshape(shapes[i])[channels].astype(numpy.float32)

    return partition


def bag_from_directory(
    *,
    path: str,
    channels: List[int],
    partition_size: int,
    gpu_accelerated: bool,
    limit: int = -1,
    regex: str,
) -> Tuple[dask.bag.Bag, dask.dataframe.DataFrame, Mapping[str, Any], int]:
    """

    Args:
        path (str): Directory to find tiffs

    Returns:
        dask.bag: bag containing dictionaries with image data
    """

    match = re.search(regex, str(path))
    groups = match.groupdict()

    z = zarr.open(path, mode="r")
    path = Path(path)
    events = []

    if limit < 0:
        limit = len(z)

    for i, obj in enumerate(z.attrs["object_number"][:limit]):
        events.append({**groups, **{
            "path": str(path),
            "zarr_idx": i,
            "object_number": obj
        }})

    bag = dask.bag.from_sequence(events, partition_size=partition_size)
    bag = bag.map_partitions(load_image_partition, z, channels)

    loader_meta = dict(path=str, zarr_idx=int, object_number=int)
    for k in groups.keys():
        loader_meta[k] = str
    return bag, loader_meta, len(events)
