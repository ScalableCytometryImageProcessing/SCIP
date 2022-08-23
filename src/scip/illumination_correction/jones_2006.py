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
    median_filter_size: int = 50,
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

        return dict(
            pixels=total2["pixels"] + total1["pixels"],
            count=total1["count"] + total2["count"]
        )

    def finish(total):
        tmp = numpy.asarray([
            median_filter(
                total[1]["pixels"][i] / total[1]["count"],
                size=median_filter_size,
                mode="constant"
            ) for i in range(len(total[1]["pixels"]))
        ])
        return (
            total[0],
            numpy.where(tmp == 0, 1, tmp)  # swap out 0 for division no-op 1
        )

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
