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
from dask.distributed import (Client, LocalCluster)
import dask.bag

from scip.utils import util


def get_images_data(n=10):
    return numpy.tile(numpy.arange(0, 100).reshape(10, 10)[numpy.newaxis], (n, 3, 1, 1))


def get_mask_data(n=10):
    return numpy.full(shape=(10, 3, 10, 10), fill_value=True, dtype=bool)


def to_records(images, masks):
    assert len(images) == len(masks)
    return [{
        "pixels": image,
        "mask": mask,
        "group": "one"
    } for image, mask in zip(images, masks)]


@pytest.fixture(scope="function")
def images_bag():
    images = get_images_data()
    masks = get_mask_data()
    records = to_records(images, masks)

    bag = dask.bag.from_sequence(records, partition_size=5)
    return bag


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
