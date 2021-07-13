# Contains methods for administrative tasks

from dask.distributed import (Client, LocalCluster)
from dask_jobqueue import PBSCluster
from pathlib import Path

def get_client(local=True, n_workers=2):
    """
    Sets up a cluster and client.

    local (bool): If true, sets up a LocalCluster, otherwise a PBSCluster
    n_workers (int): Defines the amount of workers the cluster will create
    """

    if local:
        cluster = LocalCluster(n_workers=n_workers)
    else:
        assert (Path.home() / "logs").exists(), "Make sure directory 'logs' exists in your home dir"

        cluster = PBSCluster(
            cores=24,
            memory="10GB",
            walltime=None,
            resource_spec="h_vmem=10G,mem_free=240G",
            processes=6,
            project="SIP",
            job_extra=("-pe serial 24", "-j y", "-o ~/logs/dask_workers.out")
        )
        cluster.scale(jobs=n_workers)

    return Client(cluster)
