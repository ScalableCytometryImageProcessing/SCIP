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
import pytest


@pytest.mark.parametrize(
    "fake_images_bag, expected_quantiles",
    [(True, [0., 99.]), (False, [22., 77.])],
    indirect=["fake_images_bag"]
)
def test_distributed_minmax(
    fake_images_bag,
    expected_quantiles,
    fake_image_nchannels
):
    quantiles = quantile_normalization.get_distributed_minmax(fake_images_bag, fake_image_nchannels)
    quantiles = dict(quantiles.compute())

    assert len(quantiles) == 2
    assert all(k in quantiles.keys() for k in ["one", "two"])
    assert numpy.array_equal(quantiles["one"], numpy.array(
        [expected_quantiles] * fake_image_nchannels))
    assert numpy.array_equal(quantiles["two"], numpy.array(
        [expected_quantiles] * fake_image_nchannels))


@pytest.mark.parametrize("fake_images_bag", [True], indirect=["fake_images_bag"])
def test_quantile_normalization(
    fake_images_bag,
    fake_image_nchannels
):

    images, _ = quantile_normalization.quantile_normalization(
        fake_images_bag, fake_image_nchannels)
    images = images.compute()

    assert len(images) > 0
    assert all(max(1, im["pixels"].max()) == 1 for im in images)
    assert all(min(0, im["pixels"].min()) == 0 for im in images)
