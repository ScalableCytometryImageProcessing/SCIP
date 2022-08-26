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
Data loader for Carl Zeiss Image format. Based on the aicsimageio package. Expects scenes and CZXY
dimensions to be present.
"""

import re
from typing import List, Mapping, Any

from aicsimageio import AICSImage
import dask.bag
import dask.dataframe
import dask.array
import dask

from scip.loading import util as l_util


def _load_block(event, channels):
    im = AICSImage(event["path"], reconstruct_mosaic=False)
    im.set_scene(event["scene"])

    newevent = event.copy()

    if channels is not None:
        newevent["pixels"] = im.get_image_data("CZXY", T=0, C=channels)
    else:
        newevent["pixels"] = im.get_image_data("CZXY", T=0)

    return newevent


def get_loader_meta(**kwargs) -> Mapping[str, type]:
    return dict(path=str, tile=int, scene=str, id=int)


def get_group_keys():
    return ["scene", "tile"]


@dask.delayed
def meta_from_directory(
    path: str,
    scenes: List[str]
) -> List[Mapping[str, Any]]:

    im = AICSImage(path, reconstruct_mosaic=False)

    if (scenes is None) or (type(scenes) is str):
        im_scenes = im.scenes
        if type(scenes) is str:
            im_scenes = filter(lambda s: re.match(scenes, s), im_scenes)
    elif type(scenes) is list:
        im_scenes = scenes
    else:
        raise ValueError("Scenes configuration cannot be recognized.")

    scenes_meta = []
    for scene in im_scenes:
        # store the scene and tile name
        scenes_meta.extend([dict(scene=scene, tile=i, path=path) for i in range(im.shape[0])])

    return scenes_meta


def load_pixels(
    images: dask.bag.Bag,
    channels: List[int],
    **kwargs
) -> dask.bag.Bag:
    return images.map_partitions(
        l_util._load_image_partition, channels=channels, load=_load_block)
