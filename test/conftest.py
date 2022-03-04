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

import pytest
import numpy
import dask.bag

from dask.distributed import (Client, LocalCluster)
from scip.utils import util
from scip.loading import multiframe_tiff
from scip.masking import threshold


# HELPERS

def fake_mask(image_nchannels, n=10, full=True):
    mask = numpy.full(shape=(n, image_nchannels, 10, 10), fill_value=True, dtype=bool)

    if not full:
        mask[:, :, [0, 1, -1, -2], :] = False
        mask[:, :, :, [0, 1, -1, -2]] = False

    return mask


def to_records(images, masks):
    assert len(images) == len(masks)
    return [{
        "pixels": image,
        "mask": mask,
        "combined_mask": mask[0],
        "background": [0] * len(images[0]),
        "combined_background": [0] * len(images[0]),
        "group": "one",
        "bbox": (2, 2, 8, 8),
        "regions": [1] * len(images[0])
    } for image, mask in zip(images, masks)]


# FIXTURES

@pytest.fixture(scope="session")
def fake_images(image_nchannels, n=10):
    return numpy.tile(
        numpy.arange(0, 100).reshape(10, 10)[numpy.newaxis], (n, image_nchannels, 1, 1))


@pytest.fixture(scope="session")
def fake_images_bag(fake_images, image_nchannels):
    records = to_records(fake_images, fake_mask(image_nchannels, full=True))
    bag = dask.bag.from_sequence(records, partition_size=5)
    return bag


@pytest.fixture(scope="session")
def fake_masked_images_bag(fake_images, image_nchannels):
    records = to_records(fake_images, fake_mask(image_nchannels, full=False))
    bag = dask.bag.from_sequence(records, partition_size=5)
    return bag


@pytest.fixture(scope="session")
def images_bag(images_folder):
    bag, _ = multiframe_tiff.bag_from_directory(
        images_folder, channels=[0, 1, 2], partition_size=2)
    return threshold.create_masks_on_bag(bag, main_channel=0, smooth=0.75)


@pytest.fixture(scope="session")
def image_nchannels():
    return 3


@pytest.fixture(scope="session")
def images_folder():
    return Path("test/data/images")


@pytest.fixture(scope="session")
def image_path():
    return Path("test/data/images/pbmc+PI_00000000.tiff")


@pytest.fixture(scope="session")
def data():
    return Path("test/data/")


@pytest.fixture(scope="session")
def cluster():
    cluster = LocalCluster(n_workers=1)
    client = Client(cluster)
    yield client
    client.close()
    cluster.close()


@pytest.fixture(scope="session")
def config(data):
    return util.load_yaml_config(str(data / "scip.yml"))
