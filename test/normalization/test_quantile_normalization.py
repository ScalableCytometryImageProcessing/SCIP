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

from scip.normalization import quantile_normalization
import numpy


def test_distributed_minmax(images_bag, image_nchannels):
    quantiles = quantile_normalization.get_distributed_minmax(images_bag, image_nchannels)
    quantiles = quantiles.compute()

    assert len(quantiles) == 1
    assert quantiles[0][0] == "one"
    assert numpy.array_equal(quantiles[0][1], numpy.array([[0., 99.]] * image_nchannels))


def test_masked_distributed_minmax(images_masked_bag, image_nchannels):
    quantiles = quantile_normalization.get_distributed_minmax(images_masked_bag, image_nchannels)
    quantiles = quantiles.compute()

    assert len(quantiles) == 1
    assert quantiles[0][0] == "one"
    assert numpy.array_equal(quantiles[0][1], numpy.array([[22., 77.]] * image_nchannels))
