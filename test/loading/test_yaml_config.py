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


def test_correct_amount_of_channels(images_folder, config):
    bag, _ = multiframe_tiff.bag_from_directory(
        images_folder, partition_size=2, channels=config["loading"]["channels"])
    images = bag.compute()
    assert len(images[0]["pixels"]) == len(config["loading"]["channels"])
