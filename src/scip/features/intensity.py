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
import scipy.stats
from skimage.morphology import disk, binary_erosion
from numba import njit

props = [
    'mean',
    'median',
    'max',
    'min',
    'std',
    'mad',
    'lower_quartile',
    'upper_quartile',
    'sum',
    'skewness',
    'kurtosis',
]


def _intensity_features_meta(channel_names: List[str]) -> Mapping[str, type]:
    out = {}
    for i in channel_names:
        out.update({f"{p}_{i}": float for p in props})
        out.update({f"bgcorr_{p}_{i}": float for p in props})
        out.update({f"edge_{p}_{i}": float for p in props})
        out.update({f"bgcorr_edge_{p}_{i}": float for p in props})
        out.update({f"combined_{p}_{i}": float for p in props})
        out.update({f"combined_bgcorr_{p}_{i}": float for p in props})
        out.update({f"combined_edge_{p}_{i}": float for p in props})
        out.update({f"combined_bgcorr_edge_{p}_{i}": float for p in props})
    return out


@njit(cache=True)
def _row(pixels: numpy.ndarray) -> list:
    percentiles = numpy.percentile(pixels, q=(0, 25, 50, 75, 100))

    d = [
        numpy.mean(pixels),
        percentiles[2],
        percentiles[4],
        percentiles[0],
        numpy.std(pixels),
        numpy.median(numpy.absolute(pixels - percentiles[2])),
        percentiles[1],
        percentiles[3],
        numpy.sum(pixels)
    ]

    return d


def _row2(pixels: numpy.ndarray) -> list:
    return [
        scipy.stats.skew(pixels),
        scipy.stats.kurtosis(pixels)
    ]


def _get_edge_mask(mask: numpy.ndarray) -> numpy.ndarray:
    return numpy.bitwise_xor(binary_erosion(mask, footprint=disk(6)), mask)


def intensity_features(
    pixels: numpy.ndarray,
    mask: numpy.ndarray,
    combined_mask: numpy.ndarray,
    background: numpy.ndarray,
    combined_background: numpy.ndarray,
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

    The features are computed on 8 different views on the pixel data:
        1. Raw values of channel specific mask
        2. Background substracted values of channel specific mask
        3. Edge values of channel specific mask
        4. Background substracted edge values of channel specific mask
        5. Raw values of union of masks
        6. Background substracted values of union of masks
        7. Edge values of union of masks
        8. Background substracted edge values of union of masks

    Args:
        sample (Mapping): mapping with pixels, mask, combined_mask, background and
          combined background keys.

    Returns:
        Mapping[str, Any]: extracted features
    """

    out = numpy.full(shape=(len(pixels), 8, len(props)), fill_value=None, dtype=float)

    for i in range(len(pixels)):

        # compute features on channel specific mask
        if numpy.any(mask[i]):

            mask_pixels = pixels[i][mask[i]]
            mask_bgcorr_pixels = mask_pixels - background[i]

            out[i, 0] = _row(mask_pixels) + _row2(mask_pixels)
            out[i, 1] = _row(mask_bgcorr_pixels) + _row2(mask_bgcorr_pixels)

            edge = _get_edge_mask(mask[i])
            if edge.any():
                mask_edge_pixels = pixels[i][edge]
                mask_bgcorr_edge_pixels = mask_edge_pixels - background[i]
                out[i, 2] = _row(mask_edge_pixels) + _row2(mask_edge_pixels)
                out[i, 3] = _row(mask_bgcorr_edge_pixels) + _row2(mask_bgcorr_edge_pixels)
            else:
                out[i, 2:] = 0
        else:
            # write default values
            out[i, :4] = 0

        # always compute features on combined mask (it can never be empty)
        mask_pixels = pixels[i][combined_mask]
        mask_bgcorr_pixels = mask_pixels - combined_background[i]

        out[i, 4] = _row(mask_pixels) + _row2(mask_pixels)
        out[i, 5] = _row(mask_bgcorr_pixels) + _row2(mask_bgcorr_pixels)

        combined_edge = _get_edge_mask(combined_mask)
        if combined_edge.any():
            mask_edge_pixels = pixels[i][combined_edge]
            mask_bgcorr_edge_pixels = mask_edge_pixels - combined_background[i]

            out[i, 6] = _row(mask_edge_pixels) + _row2(mask_edge_pixels)
            out[i, 7] = _row(mask_bgcorr_edge_pixels) + _row2(mask_bgcorr_edge_pixels)
        else:
            out[i, 6:] = 0

    return out.flatten()
