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

from typing import Mapping, List

import numpy
from skimage.measure import label, regionprops_table


prop_names = [
    "area",
    "convex_area",
    "eccentricity",
    "equivalent_diameter",
    "euler_number",
    "feret_diameter_max",
    "filled_area",
    "major_axis_length",
    "minor_axis_length",
    "orientation",
    "perimeter",
    "perimeter_crofton",
    "solidity",
    "extent",
    "inertia_tensor-0-0",
    "inertia_tensor-0-1",
    "inertia_tensor-1-0",
    "inertia_tensor-1-1",
    "inertia_tensor_eigvals-0",
    "inertia_tensor_eigvals-1",
    "moments-0-0",
    "moments-0-1",
    "moments-0-2",
    "moments-0-3",
    "moments-1-0",
    "moments-1-1",
    "moments-1-2",
    "moments-1-3",
    "moments-2-0",
    "moments-2-1",
    "moments-2-2",
    "moments-2-3",
    "moments-3-0",
    "moments-3-1",
    "moments-3-2",
    "moments-3-3",
    "moments_central-0-0",
    "moments_central-0-1",
    "moments_central-0-2",
    "moments_central-0-3",
    "moments_central-1-0",
    "moments_central-1-1",
    "moments_central-1-2",
    "moments_central-1-3",
    "moments_central-2-0",
    "moments_central-2-1",
    "moments_central-2-2",
    "moments_central-2-3",
    "moments_central-3-0",
    "moments_central-3-1",
    "moments_central-3-2",
    "moments_central-3-3",
    "moments_hu-0",
    "moments_hu-1",
    "moments_hu-2",
    "moments_hu-3",
    "moments_hu-4",
    "moments_hu-5",
    "moments_hu-6"
]
prop_ids = [
    "area",
    "convex_area",
    "eccentricity",
    "equivalent_diameter",
    "euler_number",
    "feret_diameter_max",
    "filled_area",
    "major_axis_length",
    "minor_axis_length",
    "orientation",
    "perimeter",
    "perimeter_crofton",
    "solidity",
    "extent",
    "inertia_tensor",
    "inertia_tensor_eigvals",
    "moments",
    "moments_central",
    "moments_hu"
]


def _shape_features_meta(channel_names: List[str]) -> Mapping[str, type]:
    out = {}
    for name in ["combined"] + channel_names:
        out.update({f"{p}_{name}": float for p in prop_names})
    return out


def _row(mask: numpy.ndarray) -> numpy.ndarray:
    label_img = label(mask)
    props = regionprops_table(label_image=label_img, properties=prop_ids, cache=False)
    return [numpy.mean(props[k]) for k in prop_names]


def shape_features(
    mask: numpy.ndarray,
    combined_mask: numpy.ndarray
) -> numpy.ndarray:
    """Extracts shape features from image.

    The shape features are extracted using :func:regionprops from scikit-image. These include
    features like eccentricity, convex area or equivalent diameter.

    Args:
        sample (Mapping[str, Any]): mapping with mask and combined mask keys.

    Returns:
        Mapping[str, Any]: extracted shape features.
    """

    out = numpy.full(shape=(len(prop_names * (len(mask) + 1)),), fill_value=None, dtype=float)
    out[:len(prop_names)] = _row(combined_mask)

    for i in range(len(mask)):
        if numpy.any(mask[i]):
            out[(i + 1) * len(prop_names):(i + 2) * len(prop_names)] = _row(mask[i])
        else:
            # setting proper default values if possible when the mask is empty
            out[(i + 1) * len(prop_names):(i + 2) * len(prop_names)] = [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None
            ]

    return out
