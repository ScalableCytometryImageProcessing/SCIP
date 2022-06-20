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

import re
from typing import Optional, List, Mapping, Union, Any
from importlib import import_module
from pathlib import Path

from aicsimageio import AICSImage
import dask.bag
import dask.dataframe
import dask.array
import dask

from scip.segmentation import util
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


def _project_block_partition(part, proj, **proj_kw):
    return [proj(p, **proj_kw) for p in part]


def get_loader_meta(**kwargs) -> Mapping[str, type]:
    return dict(path=str, tile=int, scene=str, id=int)


@dask.delayed
def _meta_from_directory(path, scenes):

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


def bag_from_directory(
    *,
    path: str,
    output: Optional[Path] = None,
    channels: List[int],
    partition_size: Optional[int] = None,
    gpu_accelerated: bool,
    scenes: Optional[Union[str, List[str]]] = None,
    segment_method: str,
    segment_kw: Mapping[str, Any],
    project_method: Optional[str] = "op",
    project_kw: Optional[Mapping[str, Any]] = {"op": "max"}
) -> dask.bag.Bag:
    """Creates a Dask Bag from one CZI file.

    This method loads in the scenes as Dask Arrays,
    performs Z-stack projection,
    and, finally, performs segmentation. The segmentation takes as input a
    Dask Array and outputs a Dask Bag.

    Args:
        path (str): Path to a CZI file.
        channels (list): Indices of channels to load.
        partition_size (int): Not applicable.
        gpu_accelerated (bool): Indicates wether segmentation is GPU accelerated.
        limit: (int): Not applicable.
        scenes (list|str|None): Names of the scenes that need to be loaded, None to load all scenes,
            or a regex pattern to filter scenes from all scenes in the input.
        segment_method (str): Name of the method used for segmentation.
        segment_kw (dict): Keywod arguments passed to segmentation method.
        project_method (Optional[str]): Name of the method used for projection.
        project_kw (dict, optional): Keyword arguments passed to projection method. Defaults to {}.

    Returns:
        Tuple[dask.bag.Bag, dict, int]: Bag of segmented objects, meta keys to dtype, 0

            Bag is a collection of dictionaries where each dictionary contains pixel data for
            one object and the meta keys (path, tile and scene).

            The final element is always 0 for this loading method, since it cannot be known
            upfront how many objects will be found by the segmentation method.
    """

    meta = _meta_from_directory(path, scenes)
    bag = dask.bag.from_delayed(meta)

    bag = bag.repartition(npartitions=100)

    bag = bag.map_partitions(
        l_util._load_image_partition, channels=channels, load=_load_block)

    if project_method is not None:
        project_block = import_module('scip.projection.%s' % project_method).project_block
        bag = bag.map_partitions(_project_block_partition, proj=project_block, **project_kw)

    bag = util.bag_from_blocks(
        blocks=bag,
        segment_kw=segment_kw,
        segment_method=segment_method,
        gpu_accelerated=gpu_accelerated,
        output=output,
        group_keys=["scene", "tile"]
    )

    return bag
