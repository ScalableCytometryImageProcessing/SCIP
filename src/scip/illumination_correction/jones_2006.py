import numpy
from scip.utils.util import copy_without
import dask.bag
import dask.delayed
from scipy.ndimage import median_filter
from pathlib import Path
import pickle
import dask.graph_manipulation


def correct(
    images: dask.bag.Bag,
    key: str,
    output: Path = None
) -> dask.bag.Bag:

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

        avg1 = total1["pixels"]
        if total1["count"] > 0:
            avg1 /= total1["count"]
        avg2 = total2["pixels"]
        if total2["count"] > 0:
            avg2 /= total2["count"]

        return dict(
            pixels=avg1 + avg2,
            count=total1["count"] + total2["count"]
        )

    def finish(total):
        return (total[0], median_filter(total[1]["pixels"] / total[1]["count"], size=5))

    def divide(x, mu):
        newevent = copy_without(x, without=["pixels"])
        newevent["pixels"] = x["pixels"] / mu[x["group"]]

        return newevent

    mean_images = images.foldby(
        key=key,
        binop=binop,
        combine=combine,
        initial=dict(pixels=None, count=0),
        combine_initial=dict(pixels=None, count=0)
    )
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
