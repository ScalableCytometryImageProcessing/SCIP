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

from scip.loading import zarr
import pytest
import dask.bag


@pytest.mark.parametrize("channels, expected_length", [(None, 7), ([0, 1], 2)])
def test_load_pixels(zarr_path, channels, expected_length):
    images = zarr.meta_from_directory(path=zarr_path, regex="(?P<name>.*)")
    images = dask.bag.from_delayed(images)
    images = zarr.load_pixels(images=images, channels=channels)
    images = images.compute()

    assert len(images) > 0
    assert all(len(im["pixels"]) == expected_length for im in images)
