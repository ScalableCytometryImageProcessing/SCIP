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
from . import intensity

props = intensity.props


def _raw_intensity_features_meta(channel_names: List[str]) -> Mapping[str, type]:
    out = {}
    for i in channel_names:
        out.update({f"{p}_{i}": float for p in props})
    return out


def raw_intensity_features(
    pixels: numpy.ndarray,
) -> numpy.ndarray:
    """Compute intensity features.

    Find following intensity features based on masked pixel values:
        * 'mean',
        * 'max',
        * 'min',
        * 'var',
        * 'mad',
        * 'skewness',
        * 'kurtosis',
        * 'sum',
        * 'modulation'

    The features are computed on raw values of channel

    Args:
        sample (Mapping): mapping with pixels

    Returns:
        Mapping[str, Any]: extracted features
    """

    out = numpy.full(shape=(len(pixels), len(props)), fill_value=None, dtype=float)

    for i in range(len(pixels)):
        out[i] = intensity._row(pixels[i].ravel()) + intensity._row2(pixels[i].ravel())

    return out.flatten()
