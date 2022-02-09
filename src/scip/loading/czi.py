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

from typing import Tuple, Optional, List, Mapping, Union, Any
from importlib import import_module

from aicsimageio import AICSImage
import numpy
import dask.bag
import dask.dataframe
import dask.array
import dask


def _load_scene(path, scene, channels):
    im = AICSImage(path, reconstruct_mosaic=False, chunk_dims=["Z", "C", "X", "Y"])
    im.set_scene(scene)
    return im.get_image_dask_data("MCZXY", T=0, C=channels)


def bag_from_directory(
    *,
    path: str,
    channels: List[int],
    partition_size: int,
    gpu_accelerated: bool,
    limit: int = -1,
    scenes: Union[str, List[str]],
    segment_method: str,
    segment_kw: Mapping[str, Any],
    project_method: Optional[str],
    project_kw: Mapping[str, Any] = {}
) -> Tuple[dask.bag.Bag, Mapping[str, type], int]:
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
        scenes (list): Names of the scenes that need to be loaded or 'all' to load all scenes.
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

    # it can not be known how many object will be found by the segmentation method
    # so it is not possible to limit the number of loaded objects
    assert limit == -1, "Limiting is not supported for CZI. (limit is set to {limit})."

    if scenes == "all":
        scenes = AICSImage(path, reconstruct_mosaic=False).scenes

    data = []
    scenes_meta = []
    for scene in scenes:
        data.append(_load_scene(path, scene, channels))

        # store the scene and tile name
        scenes_meta.extend([(scene, i) for i in range(data[-1].numblocks[0])])

    data = dask.array.concatenate(data)

    if project_method is not None:
        project_block = import_module('scip.projection.%s' % project_method).project_block
        data = data.map_blocks(
            project_block,
            **project_kw,
            drop_axis=2,  # Z dimension is on position 2
            dtype=data.dtype,
            meta=numpy.array((), dtype=data.dtype)
        )

    delayed_blocks = data.to_delayed().flatten()

    segment_block = import_module('scip.segmentation.%s' % segment_method).segment_block
    events = []
    for (scene, tile), block in zip(scenes_meta, delayed_blocks):

        # this segment operation is annotated with the cellpose resource to let the scheduler
        # know that it should only be executed on a worker that also has the cellpose resource.
        with dask.annotate(resources={"cellpose": 1}):
            e = segment_block(
                block,
                group=f"{scene}_{tile}",
                gpu_accelerated=gpu_accelerated,
                path=path,
                tile=tile,
                scene=scene,
                **segment_kw
            )
        events.append(e)

    return dask.bag.from_delayed(events), dict(path=str, tile=int, scene=str), 0
