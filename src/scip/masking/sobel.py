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

from skimage.filters import sobel
from skimage import morphology
import numpy
from scip.masking import mask_post_process


def get_mask(el):

    image = el["pixels"]
    mask = numpy.empty(shape=image.shape, dtype=bool)

    for dim in range(len(image)):

        elev_map = sobel(image[dim])
        closed = morphology.closing(elev_map, footprint=morphology.disk(2))

        segmentation = numpy.full(shape=closed.shape, fill_value=False, dtype=bool)
        segmentation[closed > numpy.quantile(closed, 0.9)] = True

        if segmentation.max() == 0:
            mask[dim] = False
        else:
            segmentation = segmentation == segmentation.max()
            mask[dim] = mask_post_process(segmentation)

    out = el.copy()
    out["mask"] = mask

    return out


def create_masks_on_bag(bag, **kwargs):

    def watershed_masking(partition):
        return [get_mask(p) for p in partition]

    bag = bag.map_partitions(watershed_masking)

    return dict(watershed=bag)
