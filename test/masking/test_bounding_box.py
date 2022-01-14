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

from scip.loading import multiframe_tiff
from scip.masking import watershed, util
from functools import partial


def test_bounding_box(images_folder, cluster):

    bag, _ = multiframe_tiff.bag_from_directory(
        images_folder, channels=[0, 1, 2], partition_size=2)
    bag = watershed.create_masks_on_bag(bag, noisy_channels=[0])
    bag = bag.filter(partial(util.mask_predicate, bbox_channel_index=0))
    bag = bag.map_partitions(util.bounding_box_partition)

    bag = bag.compute()

    for el in bag:
        bbox = el["bbox"]

        assert len(bbox) == 4
        assert all(isinstance(x, int) for x in bbox)
        assert bbox[0] < bbox[2]
        assert bbox[1] < bbox[3]
