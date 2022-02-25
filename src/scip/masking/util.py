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

from typing import Mapping, Any, List

import numpy as np
from skimage.measure import regionprops
import numpy
from skimage.morphology import remove_small_objects, label, remove_small_holes
from skimage.segmentation import expand_labels

from scip.utils.util import copy_without, check

from numba import njit


@njit(cache=True)
def _touching_border(mask):

    limit = 10

    if mask[0, :].sum() > limit:
        return True
    if mask[-1, :].sum() > limit:
        return True
    if mask[:, 0].sum() > limit:
        return True
    if mask[:, -1].sum() > limit:
        return True

    return False


def mask_predicate(s, bbox_channel_index):

    # a mask should be present in the bbox_channel_index
    if not numpy.any(s["mask"][bbox_channel_index]):
        return copy_without(s, without=["mask", "pixels"])

    # only one connected component should be found in the bbox channel
    if s["regions"][bbox_channel_index] != 1:
        return copy_without(s, without=["mask", "pixels"])

    # mask should not touch the border of the image
    if _touching_border(s["mask"][bbox_channel_index]):
        return copy_without(s, without=["mask", "pixels"])

    return s


def _regions_touching(arr):

    limit = 10

    # get all unique indices in the arr edges

    top = arr[0, :]
    bottom = arr[-1, :]
    left = arr[:, 0]
    right = arr[:, -1]
    a = numpy.concatenate((top, bottom, left, right))
    idx, counts = numpy.unique(a, return_counts=True)

    if idx[0] == 0:
        return idx[1:][counts[1:] > limit]
    else:
        return idx[counts > limit]


@check
def remove_regions_touching_border(p, bbox_channel_index):
    mask = numpy.empty_like(p["mask"])
    for i in range(len(mask)):
        if i == bbox_channel_index:
            mask[i] = p["mask"][i]
            continue

        x = label(p["mask"][i])
        indices = _regions_touching(x)
        mask[i] = p["mask"][i] * ~numpy.isin(x, indices)

    newevent = copy_without(p, without=["mask"])
    newevent["mask"] = mask

    return newevent


def remove_regions_touching_border_partition(part, bbox_channel_index):
    return [remove_regions_touching_border(p, bbox_channel_index) for p in part]


def apply_mask_partition(part, combined_indices=None):
    return [apply(p, combined_indices) for p in part]


@check
def apply(sample: Mapping[str, Any], combined_indices: List[int] = None):
    """
    Apply binary mask on every channel

    Args:
        sample: dictionary containg image data
        combined_indices: list of indices to be included in combined mask

    Returns:
        dict: dictionary including applied mask
    """

    img = sample["pixels"]
    mask = sample["mask"]

    i = numpy.s_[:] if combined_indices is None else combined_indices
    combined_mask = numpy.sum(mask[i], axis=0) > 0
    background = np.empty(shape=(len(img),), dtype=float)
    combined_background = np.empty(shape=(len(img),), dtype=float)

    # Multiply image with mask to set background to zero
    for i in range(img.shape[0]):
        if numpy.any(~mask[i]):
            background[i] = img[i][~mask[i]].mean()
        else:
            background[i] = 0
        combined_background[i] = img[i][~combined_mask].mean()

    output = sample.copy()
    output["combined_mask"] = combined_mask
    output["background"] = background.tolist()
    output["combined_background"] = combined_background.tolist()

    return output


def bounding_box_partition(part):
    return [get_bounding_box(event) for event in part]


@check
def get_bounding_box(event):
    minr, minc, maxr, maxc = event["pixels"].shape[1], event["pixels"].shape[2], 0, 0
    for mask in event["mask"]:
        if numpy.any(mask):
            b = regionprops(mask.astype(int))[0].bbox
            minr = min(b[0], minr)
            minc = min(b[1], minc)
            maxr = max(b[2], maxr)
            maxc = max(b[3], maxc)

    newevent = event.copy()
    newevent["bbox"] = [minr, minc, maxr, maxc]

    return newevent


def mask_post_process(mask):
    mask = remove_small_holes(mask, area_threshold=300)
    mask = expand_labels(label(mask), distance=1)
    mask = remove_small_objects(mask > 0, min_size=20)
    mask = label(mask)

    return mask > 0, mask.max()
