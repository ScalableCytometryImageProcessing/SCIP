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

from skimage.filters import threshold_otsu
from skimage.segmentation import watershed, expand_labels
from skimage.restoration import denoise_nl_means
from skimage import feature, measure
import skimage
from scipy.ndimage import distance_transform_edt

from typing import List, Any, Mapping
import numpy


def segment_block(
    events: List[Mapping[str, Any]],
    *,
    cell_diameter: int,
    dapi_channel_index: int,
    expansion_factor: float = 0.1
) -> List[dict]:

    if len(events) == 0:
        return events

    for event in events:
        plane = event["pixels"][dapi_channel_index]
        plane = skimage.img_as_float32(plane)

        plane = denoise_nl_means(plane, patch_size=3, patch_distance=2)

        t = threshold_otsu(plane)
        cells = plane > t
        distance = distance_transform_edt(cells)

        local_max_coords = feature.peak_local_max(distance, min_distance=cell_diameter)
        local_max_mask = numpy.zeros(distance.shape, dtype=bool)
        local_max_mask[tuple(local_max_coords.T)] = True
        markers = measure.label(local_max_mask)

        segmented_cells = watershed(-distance, markers)
        expanded = expand_labels(segmented_cells, distance=cell_diameter * expansion_factor)

        event["mask"] = numpy.empty(shape=event["pixels"].shape, dtype=int)
        event["mask"][dapi_channel_index] = segmented_cells
        event["mask"][[i for i in range(len(event["mask"])) if i != dapi_channel_index]] = expanded

    return events
