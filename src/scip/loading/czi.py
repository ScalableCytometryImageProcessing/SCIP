from typing import Tuple, Optional
from aicsimageio import AICSImage
import numpy
import dask.bag
import dask.dataframe
import dask.array
import dask
from importlib import import_module


def bag_from_directory(
    *,
    path: str,
    channels: list,
    partition_size: int,
    gpu_accelerated: bool,
    clip: int,
    scenes: list,
    segment_method: str,
    segment_kw: dict,
    project_method: Optional[str],
    project_kw: dict = {}
) -> Tuple[dask.bag.Bag, dict, int]:
    """Creates a Dask bag from a directory of CZI files.

    Args:
        path (str): [description]
        channels (list): [description]
        partition_size (int): [description]
        gpu_accelerated (bool): [description]
        clip (int): [description]
        scenes (list): [description]
        segment_method (str): [description]
        segment_kw (dict): [description]
        project_method (Optional[str]): [description]
        project_kw (dict, optional): [description]. Defaults to {}.

    Returns:
        Tuple[dask.bag.Bag, dict, int]: [description]
    """

    def load_scene(scene):
        im = AICSImage(path, reconstruct_mosaic=False, chunk_dims=["Z", "C", "X", "Y"])
        im.set_scene(scene)
        return im.get_image_dask_data("MCZXY", T=0, C=channels)

    if scenes == "all":
        scenes = AICSImage(path, reconstruct_mosaic=False).scenes

    data = []
    scenes_meta = []
    for scene in scenes:
        data.append(load_scene(scene))
        scenes_meta.extend([(scene, i) for i in range(data[-1].numblocks[0])])
    data = dask.array.concatenate(data)

    if project_method is not None:
        project_block = import_module('scip.projection.%s' % project_method).project_block
        data = data.map_blocks(
            project_block,
            **project_kw,
            drop_axis=2,
            dtype=data.dtype,
            meta=numpy.array((), dtype=data.dtype)
        )

    delayed_blocks = data.to_delayed().flatten()

    segment_block = import_module('scip.segmentation.%s' % segment_method).segment_block
    events = []
    for (scene, tile), block in zip(scenes_meta, delayed_blocks):
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

    return (
        dask.bag.from_delayed(events),
        dict(path=str, tile=int, scene=str),
        clip
    )
