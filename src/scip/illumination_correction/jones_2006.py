from functools import partial
import numpy
from scip.utils.util import copy_without
import dask.bag
import dask.delayed
from scipy.signal import medfilt2d
from scipy.ndimage import median_filter
from pathlib import Path
import pickle
import dask.graph_manipulation
from skimage.transform import downscale_local_mean, rescale


def correct(
    *,
    images: dask.bag.Bag,
    key: str,
    nbatches: int,
    median_filter_size: int = 50,
    downscale: int = 1,
    output: Path = None,
) -> dask.bag.Bag:

    # switch to medfilt2d for larger filter sizes as it consumes less memory
    if median_filter_size > 150:
        filter_func = partial(medfilt2d, kernel_size=median_filter_size)
    else:
        filter_func = partial(median_filter, size=median_filter_size)

    def binop(total, x):
        if total["pixels"] is None:
            total["pixels"] = numpy.zeros_like(x["pixels"])

        return dict(
            pixels=total["pixels"] + x["pixels"],
            count=total["count"] + 1
        )

    def combine(total1, total2):
        if total1["pixels"] is None:
            total1["pixels"] = numpy.zeros_like(total2["pixels"])

        return dict(
            pixels=total2["pixels"] + total1["pixels"],
            count=total1["count"] + total2["count"]
        )

    def finish(total):
        avg = total[1]["pixels"] / total[1]["count"]

        # downscale the image prior to median filtering to reduce memory consumption
        avg = downscale_local_mean(avg, factors=(1, downscale, downscale))

        avg = numpy.asarray([
            filter_func(avg[i])
            for i in range(len(total[1]["pixels"]))
        ])
        avg = numpy.where(avg == 0, 1, avg)  # swap out 0 for division no-op 1

        avg = rescale(avg, scale=downscale, anti_aliasing=True, channel_axis=0)

        return (total[0], avg)

    def divide(x, mu):
        newevent = copy_without(x, without=["pixels"])
        newevent["pixels"] = x["pixels"] / mu[x[key]]
        return newevent

    mean_images = images.foldby(
        key=key,
        binop=binop,
        combine=combine,
        initial=dict(pixels=None, count=0),
        combine_initial=dict(pixels=None, count=0)
    )
    mean_images = mean_images.repartition(npartitions=nbatches)
    mean_images = mean_images.map(finish)
    mean_images = dask.delayed(dict, pure=True)(mean_images)

    images = images.map(divide, mu=mean_images)

    if output is not None:
        @dask.delayed
        def save(mean_images):
            with open(str(output / "correction_images.pickle"), "wb") as fh:
                pickle.dump(mean_images, fh)
        return dask.graph_manipulation.bind(
            children=images, parents=save(mean_images), omit=mean_images)

    return images
