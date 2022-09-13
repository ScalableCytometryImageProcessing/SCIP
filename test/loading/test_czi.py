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

import os
import pytest
import dask.bag
from scip.loading import czi
from scip.segmentation import export_labeled_mask
from scip.projection import op, project_block_partition


@pytest.mark.parametrize("channels, expected_length", [(None, 7), ([0, 6], 2)])
def test_load_pixels(czi_path, channels, expected_length):
    images = czi.meta_from_directory(path=czi_path, scenes=None)
    images = dask.bag.from_delayed(images)
    images = czi.load_pixels(images=images, channels=channels)
    images = images.compute()

    assert len(images) > 0
    assert all(len(im["pixels"]) == expected_length for im in images)


@pytest.mark.parametrize("projection", ["mean", "max"])
def test_project(czi_path, projection):
    images = czi.meta_from_directory(path=czi_path, scenes=None)
    images = dask.bag.from_delayed(images)
    images = czi.load_pixels(images=images, channels=[0, 6])
    images = images.map(op.project_block, op=projection)
    images = images.compute()

    assert len(images) > 0
    assert all(len(im["pixels"].shape) == 3 for im in images)


@pytest.mark.skipif(
    "GITHUB_ACTIONS" in os.environ,
    reason="Bug in CellPose package related to CPNet on CPU"
)
def test_segment(czi_path, tmp_path):
    cellpose = pytest.importorskip("scip.segmentation.cellpose")

    images = czi.meta_from_directory(path=czi_path, scenes=None)
    images = dask.bag.from_delayed(images)
    images = czi.load_pixels(images=images, channels=[0, 6])
    images = images.map_partitions(project_block_partition, proj=op.project_block, op="max")

    segment_kw = dict(
        cell_diameter=30, parent_channel_index=1, dapi_channel_index=0)
    images = images.map_partitions(
        cellpose.segment_block,
        gpu_accelerated=False,
        **segment_kw
    )
    images = images.map_partitions(
        export_labeled_mask, out=tmp_path, group_keys=["scene", "tile"])

    images = images.compute()

    assert len(images) > 0
    assert all("mask" in im for im in images)
    assert all(im["mask"].any() for im in images)
    assert (tmp_path / "masks").exists()
    assert len([f for f in (tmp_path / "masks").iterdir()]) == 1


@pytest.mark.skipif(
    "GITHUB_ACTIONS" in os.environ,
    reason="Bug in CellPose package related to CPNet on CPU"
)
def test_segment_to_events(czi_path):
    cellpose = pytest.importorskip("scip.segmentation.cellpose")

    images = czi.meta_from_directory(path=czi_path, scenes=None)
    images = dask.bag.from_delayed(images)
    images = czi.load_pixels(images=images, channels=[0, 6])
    images = images.map_partitions(project_block_partition, proj=op.project_block, op="max")

    segment_kw = dict(
        cell_diameter=30, parent_channel_index=1, dapi_channel_index=0)
    images = images.map_partitions(
        cellpose.segment_block,
        gpu_accelerated=False,
        **segment_kw
    )
    images = images.map_partitions(
        cellpose.to_events,
        group_keys=["scene", "tile"], **segment_kw
    )
    images = images.compute()

    assert len(images) > 0
    assert all("pixels" in im for im in images)
