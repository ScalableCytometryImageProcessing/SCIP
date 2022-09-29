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
from scip.utils.util import check
from skimage.morphology import white_tophat, disk, label


@check
def get_mask(el, spotsize):

    regions = [0] * len(el["pixels"])
    mask = numpy.full(shape=el["pixels"].shape, dtype=bool, fill_value=False)

    # load over channels, starting with main_channel
    for dim in numpy.arange(len(el["pixels"])):
        cc = 0

        if el["mask_filter"][dim]:

            x = el["pixels"][dim]
            x = white_tophat(x, footprint=disk(spotsize))

            q = numpy.quantile(x, q=0.99)
            x = x > q
            x = label(x)
            mask[dim], cc = x > 0, x.max()

        regions[dim] = cc

    out = el.copy()
    out["mask"] = mask
    out["regions"] = regions

    return out


def create_masks_on_bag(bag, spotsize):

    def spot_masking(partition):
        return [
            get_mask(p, spotsize)
            for p in partition
        ]

    bag = bag.map_partitions(spot_masking)
    return bag
