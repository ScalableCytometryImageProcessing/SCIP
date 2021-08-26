# Contains methods for administrative tasks

from dask.distributed import (Client, LocalCluster)
from dask_jobqueue import PBSCluster
from pathlib import Path
import yaml
from pkg_resources import resource_stream
import logging
import time


class ClientClusterContext:

    def __init__(self, local=True, n_workers=2, n_processes=12, port=8787):
        """
        Sets up a cluster and client.

        local (bool): If true, sets up a LocalCluster, otherwise a PBSCluster
        n_workers (int): Defines the amount of workers the cluster will create
        """
        self.local = local
        self.n_workers = n_workers
        self.port = port
        self.n_processes = n_processes

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
                processes=self.n_processes,
                project="scip",
                job_extra=(
                    "-pe serial 24", 
                    "-j y", 
                    "-o ~/logs/dask_workers_%s.out" % str(int(time.time()*100))
                ),
                scheduler_options={
                    'dashboard_address': None if self.port is None else f':{self.port}'}
            )
            self.cluster.scale(jobs=self.n_workers)

        self.client = Client(self.cluster)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.client.close()
        self.cluster.close()


def load_yaml_config(path):
    with open(path) as fh:
        return yaml.load(fh, Loader=yaml.FullLoader)


def configure_logging(output):
    with resource_stream(__name__, 'logging.yml') as stream:
        loggingConfig = yaml.load(stream, Loader=yaml.FullLoader)

    for k,v in loggingConfig["handlers"].items():
        if k == "file":
            v["filename"] = str(output / v["filename"])

    logging.config.dictConfig(loggingConfig)
