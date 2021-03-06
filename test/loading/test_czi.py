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

from scip.loading import czi
import pytest
import os


@pytest.mark.skipif(
    "GITHUB_ACTIONS" in os.environ,
    reason="Bug in CellPose package related to CPNet on CPU"
)
@pytest.mark.parametrize("channels, expected_length", [(None, 7), ([0, 6], 2)])
def test_bag_from_directory(czi_path, channels, expected_length):
    bag = czi.bag_from_directory(
        path=czi_path,
        channels=channels,
        gpu_accelerated=False,
        segment_method="cellpose",
        segment_kw=dict(
            cell_diameter=0, segmentation_channel_index=1, dapi_channel_index=0, export=False)
    )
    images = bag.compute()

    assert all(len(im["pixels"]) == expected_length for im in images)


def test_export_mask(czi_path, tmp_path):
    bag = czi.bag_from_directory(
        path=czi_path,
        channels=[0, 6],
        gpu_accelerated=False,
        segment_method="cellpose",
        output=tmp_path,
        segment_kw=dict(
            cell_diameter=0, segmentation_channel_index=1, dapi_channel_index=0,
            export=True)
    )

    bag.compute()

    assert (tmp_path / "masks").exists()
    assert len([f for f in (tmp_path / "masks").iterdir()]) == 1
