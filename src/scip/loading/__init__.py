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
Data loaders for various image formats. All data loaders take a path as input and produce a
lazy collection containing all fields of view with meta data.
"""

from typing import List, Mapping, Any, Tuple

import dask.bag
from pathlib import Path


def load_meta(
    *,
    paths: List[str],
    kwargs: Mapping[str, Any] = {},
    loader_module
) -> Tuple[dask.bag.Bag, Mapping[str, type]]:

    bags = []
    for path in paths:
        assert Path(path).exists(), f"{path} does not exist."

        meta = loader_module.meta_from_directory(path=path, **kwargs)
        bag = dask.bag.from_delayed(meta)
        bags.append(bag)

    return dask.bag.concat(bags)


def load_pixels(
    *,
    bag: dask.bag.Bag,
    channels: List[int],
    kwargs: Mapping[str, Any] = {},
    loader_module
) -> dask.bag.Bag:

    bag = loader_module.load_pixels(bag, channels=channels, **kwargs)

    return bag
