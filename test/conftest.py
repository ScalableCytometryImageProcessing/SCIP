import pytest
from pathlib import Path
from dask.distributed import (Client, LocalCluster)
from scip.utils import util


@pytest.fixture(scope="session")
def data():
    # return Path("test/data")
    return Path("/home/maximl/shared_scratch/vulcan_pbmc_debug")


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
