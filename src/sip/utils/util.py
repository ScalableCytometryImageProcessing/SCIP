# Contains methods for administrative tasks

from dask.distributed import (Client, LocalCluster)
from dask_jobqueue import PBSCluster
from pathlib import Path
import yaml


class ClientClusterContext:

    def __init__(self, local=True, n_workers=2, port=8787):
        """
        Sets up a cluster and client.

        local (bool): If true, sets up a LocalCluster, otherwise a PBSCluster
        n_workers (int): Defines the amount of workers the cluster will create
        """
        self.local = local
        self.n_workers = n_workers
        self.port = port

    def __enter__(self):
        if self.local:
            self.cluster = LocalCluster(n_workers=self.n_workers)
        else:
            assert (Path.home() / "logs").exists(), "Make sure directory\
                 'logs' exists in your home dir"

            self.cluster = PBSCluster(
                cores=24,
                memory="240GiB",
                resource_spec="h_vmem=10G,mem_free=240G",
                processes=12,
                project="SIP",
                job_extra=("-pe serial 24", "-j y", "-o ~/logs/dask_workers.out"),
                scheduler_options={'dashboard_address': f':{self.port}'}
            )
            self.cluster.scale(jobs=self.n_workers)

        self.client = Client(self.cluster)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.client.close()
        self.cluster.close()


def load_yaml_config(path):
    with open(path) as fh:
        return yaml.load(fh)
