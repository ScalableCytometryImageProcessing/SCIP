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

import numpy
from scipy.stats import normaltest
from scip.utils.util import check, copy_without
from scip.masking import mask_predicate
from skimage.morphology import white_tophat, disk, label, binary_dilation
from skimage.filters import threshold_minimum


@check
def get_mask(el, main_channel, spotsize):

    regions = [0] * len(el["pixels"])
    mask = numpy.full(shape=el["pixels"].shape, dtype=bool, fill_value=False)

    # load over channels, starting with main_channel
    arr = numpy.arange(len(el["pixels"]))
    for dim in [main_channel] + numpy.delete(arr, arr == main_channel).tolist():
        cc = 0

        x = el["pixels"][dim]
        if (normaltest(x.ravel()).pvalue < 0.05):

            x = el["pixels"][dim]
            x = white_tophat(x, footprint=disk(spotsize))

            for nbins in [256, 512, 1024]:
                try:
                    x = threshold_minimum(x, nbins=nbins) < x
                    x = binary_dilation(x, footprint=disk(2))
                    x = label(x)
                    mask[dim], cc = x > 0, x.max()
                    break
                except RuntimeError:
                    pass
        elif dim == main_channel:
            out = copy_without(el, without=["pixels"])
            out["regions"] = regions
            out["mask"] = mask
            return out

        regions[dim] = cc

    out = el.copy()
    out["mask"] = mask
    out["regions"] = regions

    return out


def create_masks_on_bag(bag, main_channel, spotsize):

    def spot_masking(partition):
        return [
            mask_predicate(get_mask(p, main_channel, spotsize), main_channel)
            for p in partition
        ]

    bag = bag.map_partitions(spot_masking)
    return bag
