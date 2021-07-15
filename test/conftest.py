import pytest
from pathlib import Path
from dask.distributed import (Client, LocalCluster)


@pytest.fixture(scope="session")
def data():
    return Path("test/data")


@pytest.fixture(scope="session")
def cluster():
    cluster = LocalCluster(n_workers=1)
    client = Client(cluster)
    yield client
    client.close()
    cluster.close()
