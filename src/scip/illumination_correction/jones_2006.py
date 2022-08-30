"""
Illumination correction as proposed by Jones et al. (2006)
"""

from functools import partial
import numpy
import dask.bag
import dask.delayed
from scipy.signal import medfilt2d
from scipy.ndimage import median_filter
from pathlib import Path
import pickle
import dask.graph_manipulation
from skimage.transform import downscale_local_mean, rescale
import copy


def _binop(total, x):
    if total["pixels"] is None:
        total["pixels"] = numpy.zeros_like(x["pixels"])

    return dict(
        pixels=total["pixels"] + x["pixels"],
        count=total["count"] + 1
    )


def _combine(total1, total2):
    if total1["pixels"] is None:
        total1["pixels"] = numpy.zeros_like(total2["pixels"])

    return dict(
        pixels=total2["pixels"] + total1["pixels"],
        count=total1["count"] + total2["count"]
    )


def _finish(total, filter_func, downscale):
    avg = total[1]["pixels"] / total[1]["count"]

    if downscale > 1:
        # downscale the image prior to median filtering to reduce memory consumption
        avg = downscale_local_mean(avg, factors=(1, downscale, downscale))

    avg = numpy.asarray([
        filter_func(avg[i])
        for i in range(len(total[1]["pixels"]))
    ])
    avg = numpy.where(avg == 0, 1, avg)  # swap out 0 for division no-op 1

    if downscale > 1:
        # reverse earlier downscaling
        avg = rescale(avg, scale=downscale, anti_aliasing=True, channel_axis=0)

    return (total[0], avg)


def correct(
    *,
    images: dask.bag.Bag,
    key: str,
    ngroups: int,
    median_filter_size: int = 50,
    downscale: int = 1,
    output: Path = None,
    precomputed: Path = None
) -> dask.bag.Bag:
    """
    Distributed implementation of Jones et al. (2006) illumination correction. All images
    are averaged per batch, after which the image is filtered using a median filter. If requested,
    the image is downscaled prior to median filtering to reduce memory consumption.

    Args:
        images: Collection containing images to be corrected. Each item in the collection a
            pixels and 'key' key.
        key: Item key used for grouping.
        ngroups: Number of groups in the images collection.
        median_filter_size: Size of the window used in the median filter.
        downscale: factor by which to downscale the image prior to median filtering
        output: Path pointing to directory to save correction images.
        precomputed: Path to pickle file with precomputed correction images in a dict.

    Returns:
        Collection with corrected images.
    """

    # switch to medfilt2d for larger filter sizes as it consumes less memory
    if median_filter_size > 150:
        filter_func = partial(medfilt2d, kernel_size=median_filter_size)
    else:
        filter_func = partial(median_filter, size=median_filter_size)

    def divide(part, mu):
        newpart = copy.deepcopy(part)
        for x in newpart:
            x["pixels"] = x["pixels"] / mu[x[key]]
        return newpart

    if precomputed is not None:
        @dask.delayed(pure=True)
        def load_images(p):
            with open(p, "rb") as fh:
                return pickle.load(fh)
        mean_images = load_images(str(precomputed))
    else:
        mean_images = images.foldby(
            key=key,
            binop=_binop,
            combine=_combine,
            initial=dict(pixels=None, count=0),
            combine_initial=dict(pixels=None, count=0)
        )
        mean_images = mean_images.repartition(npartitions=ngroups)
        mean_images = mean_images.map(_finish, filter_func=filter_func, downscale=downscale)
        mean_images = dask.delayed(dict, pure=True)(mean_images)

    images = images.map_partitions(divide, mu=mean_images)

    if output is not None:
        @dask.delayed
        def save(mean_images):
            with open(str(output / "correction_images.pickle"), "wb") as fh:
                pickle.dump(mean_images, fh)
        return dask.graph_manipulation.bind(
            children=images, parents=save(mean_images), omit=mean_images)

    return images
