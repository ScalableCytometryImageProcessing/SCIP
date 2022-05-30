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

from typing import Mapping, List, Optional

import dask
import dask.bag
import dask.dataframe
import zarr
import numpy
from pathlib import Path
import re
import copy


def reload_image_partition(
    partition: List,
    channels: List[int],
    regex: str
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


def get_loader_meta(
    *,
    regex: str,
    **kwargs
) -> Mapping[str, type]:
    loader_meta = dict(path=str, zarr_idx=int, object_number=int)
    named_groups = re.findall(r"\(\?P\<([^>]+)\>[^)]+\)", regex)
    for k in named_groups:
        loader_meta[k] = str
    return loader_meta


def bag_from_directory(
    *,
    path: str,
    output: Optional[Path] = None,
    channels: List[int],
    partition_size: int,
    gpu_accelerated: bool,
    regex: str
) -> dask.bag.Bag:
    """

    Args:
        path (str): Directory to find tiffs

    Returns:
        dask.bag: bag containing dictionaries with image data
    """

    match = re.search(regex, str(path))
    groups = match.groupdict()

    z = zarr.open(path, mode="r")

    if channels is None:
        channels = numpy.s_[:]

    path = Path(path)
    events = []

    for i, obj in enumerate(z.attrs["object_number"]):
        events.append({**groups, **{
            "path": str(path),
            "zarr_idx": i,
            "object_number": obj
        }})

    bag = dask.bag.from_sequence(events, partition_size=partition_size)
    bag = bag.map_partitions(load_image_partition, z, channels)

    return bag
