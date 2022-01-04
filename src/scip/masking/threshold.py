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
from skimage.morphology import (
    closing, disk, remove_small_holes, remove_small_objects, label)
from skimage.filters import threshold_otsu, sobel, gaussian
from skimage.segmentation import expand_labels
from scipy.stats import normaltest
from scip.utils.util import check


@check
def get_mask(el, main, main_channel):

    if main:
        regions = [0] * len(el["pixels"])
        mask, cc = numpy.full(shape=el["pixels"].shape, dtype=bool, fill_value=False), 0
        x = el["pixels"][main_channel]
        if (normaltest(x.ravel()).pvalue < 0.05):
            x = sobel(x)
            x = closing(x, selem=disk(2))
            x = threshold_otsu(x) < x
            x = remove_small_holes(x, area_threshold=100)
            x = remove_small_objects(x, min_size=20)
            x = label(x)
            x = expand_labels(x, distance=1)
            mask[main_channel], cc = x > 0, x.max()
        regions[main_channel] = cc
    else:
        regions = []
        # search for objects within the bounding box found on the main_channel
        mask = el["mask"]
        bbox = el["bbox"]
        for dim in range(len(el["pixels"])):
            if dim == main_channel:
                # in this phase the main channel always has 1 component
                regions.append(1)
                continue

            x = el["pixels"][dim, bbox[0]:bbox[2], bbox[1]:bbox[3]]
            x = gaussian(x, sigma=1)
            x = threshold_otsu(x) < x
            x[[0, -1], :] = 0
            x[:, [0, -1]] = 0
            x = remove_small_holes(x, area_threshold=10)
            x = remove_small_objects(x, min_size=5)
            x = label(x)
            mask[dim, bbox[0]:bbox[2], bbox[1]:bbox[3]], cc = x > 0, x.max()
            regions.append(cc)

    out = el.copy()
    out["mask"] = mask
    out["regions"] = regions

    return out


def create_masks_on_bag(bag, main, main_channel):

    def threshold_masking(partition):
        return [get_mask(p, main, main_channel) for p in partition]

    bag = bag.map_partitions(threshold_masking)
    return bag
