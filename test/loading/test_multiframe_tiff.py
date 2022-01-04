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

from pathlib import Path
from scip.loading import multiframe_tiff


def test_load_image(image_path: Path):
    event = dict(path=image_path)
    im = multiframe_tiff.load_image(event)

    assert "path" in im
    assert "pixels" in im
    assert str(im["path"]) == str(event["path"])
    assert im["pixels"].mean() > 0


def test_bag_from_directory(images_folder, cluster):
    bag, meta = multiframe_tiff.bag_from_directory(
        images_folder, channels=None, partition_size=2)
    images = bag.compute()
    assert len(images) == 11
    assert len(images) == len(meta)
    assert all(len(im["pixels"]) == 8 for im in images)
