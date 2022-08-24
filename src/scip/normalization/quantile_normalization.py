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

import numpy as np
import dask
import dask.bag
import dask.array
from scip.utils.util import check, copy_without


def get_distributed_minmax(bag, nchannels):  # noqa: C901

    def combine_extent_partition(a, b):

        if "pixels" not in b:
            return a

        if "mask" in b:
            b = [b["pixels"][i][b["mask"][i]] for i in range(nchannels)]
        else:
            b = b["pixels"]

        out = np.empty(shape=a.shape)
        for i in range(nchannels):
            if b[i].size == 0:
                out[i] = a[i]
            else:
                out[i, 0] = min(a[i, 0], np.min(b[i]))
                out[i, 1] = max(a[i, 1], np.max(b[i]))
        return out

    def final_minmax(a, b):
        out = np.empty(shape=a.shape)
        for i in range(nchannels):
            out[i, 0] = min(a[i, 0], b[i, 0])
            out[i, 1] = max(a[i, 1], b[i, 1])
        return out

    init = np.empty(shape=(nchannels, 2))
    init[:, 0] = np.inf
    init[:, 1] = -np.inf
    out = bag.foldby(
        key="group",
        binop=combine_extent_partition,
        combine=final_minmax,
        initial=init,
        combine_initial=init
    )

    return out


@check
def sample_normalization(sample, quantiles):
    """
    Perform min-max normalization using quantiles on original pixel data,
    masked pixel data and flat masked intensities list

    Args:
        sample (dict): dictionary containing image data and mask data
        qq: (lower, upper) list of quantiles for every channel
    Returns:
        dict: dictionary including normalized data
    """

    qq = dict(quantiles)[sample["group"]]

    pixels = np.empty_like(sample["pixels"])
    for i in range(len(sample["pixels"])):
        pixels[i] = (sample["pixels"][i] - qq[i, 0]) / (qq[i, 1] - qq[i, 0])

    newsample = copy_without(sample, without="pixels")
    newsample["pixels"] = pixels

    return newsample


def quantile_normalization(images: dask.bag.Bag, nchannels):
    """
    Apply min-max normalization on all images, both on original pixel data and masked pixel data

    Args:
        images (dask.bag): bag of dictionaries containing image data
    Returns:
        dask.bag: bag of dictionaries including normalized data
    """

    def normalize_partition(part, quantiles):
        return [sample_normalization(p, quantiles) for p in part]

    quantiles = get_distributed_minmax(images, nchannels)
    images = images.map_partitions(
        normalize_partition, quantiles.to_delayed()[0])

    return images, quantiles
