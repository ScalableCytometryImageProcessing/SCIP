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


def load_image(event, z, channels, clip):
    """
    Load an image from a certain path

    Args:
        path (str): path of image,
        z (zarr): zarr object
        channels (list, optional): image channels to load. Defaults to None.

    Returns:
        dict: dictionary containing pixel values (ndarray) and path for each image
    """

    i = event["zarr_idx"]
    if clip is not None:
        event["pixels"] = numpy.clip(z[i].reshape(z.attrs["shape"][i])[channels], 0, clip)
    else:
        event["pixels"] = z[i].reshape(z.attrs["shape"][i])[channels]
    event["pixels"] = event["pixels"].astype(numpy.float32)
    return event


def load_image_partition(partition, z, channels, clip):
    return [load_image(event, z, channels, clip) for event in partition]


def bag_from_directory(
    *,
    path: str,
    channels: list,
    partition_size: int,
    gpu_accelerated: bool,
    clip: int,
    regex: str,
    limit: int = -1
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
    return bag, loader_meta, clip
