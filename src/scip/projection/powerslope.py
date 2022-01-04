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
from centrosome import radial_power_spectrum
import scipy.linalg


def compute_powerslope(pixel_data):
    radii, magnitude, power = radial_power_spectrum.rps(pixel_data)
    if sum(magnitude) > 0 and len(numpy.unique(pixel_data)) > 1:
        valid = magnitude > 0
        radii = radii[valid].reshape((-1, 1))
        power = power[valid].reshape((-1, 1))
        if radii.shape[0] > 1:
            idx = numpy.isfinite(numpy.log(power))
            powerslope = scipy.linalg.basic.lstsq(
                numpy.hstack(
                    (
                        numpy.log(radii)[idx][:, numpy.newaxis],
                        numpy.ones(radii.shape)[idx][:, numpy.newaxis],
                    )
                ),
                numpy.log(power)[idx][:, numpy.newaxis],
            )[0][0]
        else:
            powerslope = 0
    else:
        powerslope = 0
    return powerslope


def project_block(block: numpy.ndarray) -> numpy.ndarray:
    scores = numpy.empty(shape=block.shape[:3], dtype=float)
    for m, c, z in numpy.ndindex(block.shape[:3]):
        slope = compute_powerslope(block[m, c, z])
        if hasattr(slope, "shape"):
            scores[m, c, z] = slope[0]
        else:
            scores[m, c, z] = slope

    indices = numpy.squeeze(scores.argmax(axis=2))
    return numpy.vstack([block[:, i, j] for i, j in enumerate(indices)])[numpy.newaxis]
