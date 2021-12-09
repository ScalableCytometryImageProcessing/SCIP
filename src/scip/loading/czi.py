from aicsimageio import AICSImage
from typing import Tuple, Callable
from dask.dataframe.core import repartition
import numpy
import dask.bag
import dask.dataframe
import dask.array
import dask
from centrosome import radial_power_spectrum
import scipy.linalg
import pandas
from importlib import import_module


def compute_powerslope(pixel_data):
    radii, magnitude, power = radial_power_spectrum.rps(pixel_data)
    if sum(magnitude) > 0 and len(numpy.unique(pixel_data)) > 1:
        valid = magnitude > 0
        radii = radii[valid].reshape((-1, 1))
        power = power[valid].reshape((-1, 1))
        if radii.shape[0] > 1:
            idx = numpy.isfinite(numpy.log(power))
            powerslope = scipy.linalg.basic.lstsq(
                numpy.hstack(
                    (
                        numpy.log(radii)[idx][:, numpy.newaxis],
                        numpy.ones(radii.shape)[idx][:, numpy.newaxis],
                    )
                ),
                numpy.log(power)[idx][:, numpy.newaxis],
            )[0][0]
        else:
            powerslope = 0
    else:
        powerslope = 0
    return powerslope


def select_focused_plane(block):
    scores = numpy.empty(shape=block.shape[:3], dtype=float)
    for m, c, z in numpy.ndindex(block.shape[:3]):
        slope = compute_powerslope(block[m, c, z])
        if hasattr(slope, "shape"):
            scores[m, c, z] = slope[0]
        else:
            scores[m, c, z] = slope

    indices = numpy.squeeze(scores.argmax(axis=2))
    return numpy.vstack([block[:, i, j] for i, j in enumerate(indices)])[numpy.newaxis]


def bag_from_directory(
    *,
    path: str,
    channels: list,
    partition_size: int,
    gpu_accelerated: bool,
    clip: int,
    scenes: list,
    segment_method: str,
    segment_kw: dict
) -> Tuple[dask.bag.Bag, dict, int]:

    segment_block = import_module('scip.segmentation.%s' % segment_method).segment_block

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

    data = data.map_blocks(
        select_focused_plane,
        drop_axis=2,
        dtype=data.dtype,
        meta=numpy.array((), dtype=data.dtype)
    )

    delayed_blocks = data.to_delayed().flatten()

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
