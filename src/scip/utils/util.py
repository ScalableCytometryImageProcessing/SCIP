# Contains methods for administrative tasks

from dask.distributed import (Client, LocalCluster)
from dask_jobqueue import PBSCluster
from pathlib import Path
import yaml
from pkg_resources import resource_stream
import logging
import time
import math
from dask.distributed import core


class ClientClusterContext:

    def __init__(
            self,
            *,
            local=True,
            n_workers=2,
            n_processes=12,
            port=8787,
            local_directory=None,
            memory=None,
            cores=None,
            job_extra=[],
            walltime="01:00:00"
    ):
        """
        Sets up a cluster and client.

        local (bool): If true, sets up a LocalCluster, otherwise a PBSCluster
        n_workers (int): Defines the amount of workers the cluster will create
        """
        self.local = local
        self.n_workers = n_workers
        self.port = port
        self.n_processes = n_processes
        self.local_directory = local_directory
        self.memory = memory
        self.cores = cores
        self.job_extra = job_extra
        self.walltime = walltime

    def __enter__(self):
        if self.local:
            self.cluster = LocalCluster(n_workers=self.n_workers)
        else:
            assert (Path.home() / "logs").exists(), "Make sure directory\
                 'logs' exists in your home dir"

            nodes_needed = int(math.ceil(self.n_workers / self.n_processes))
            mb_needed = math.ceil(self.memory * 1073.74)
            self.cluster = PBSCluster(
                cores=self.cores,
                memory=f"{self.memory}GiB",
                resource_spec=f"nodes={nodes_needed}:ppn={self.cores},mem={mb_needed}mb",
                processes=self.n_processes,
                project=None,
                local_directory=self.local_directory,
                walltime=self.walltime,
                job_extra=self.job_extra,
                scheduler_options={
                    'dashboard_address': None if self.port is None else f':{self.port}'},
            )

            self.cluster.scale(jobs=self.n_workers)

        self.client = Client(self.cluster)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):

        if exc_type is not None:
            logging.getLogger(__name__).error(
                "Exception in context: %s, %s", exc_type, str(exc_value))
            logging.getLogger(__name__).error(exc_traceback)

        self.client.close()
        self.cluster.close()

    def wait(self):
        while True:
            running = 0 
            for w in self.cluster.workers:
                running += (w.status == core.Status.running)

            if running != self.n_workers:
                time.sleep(1)
            else:
                break


def load_yaml_config(path):
    with open(path) as fh:
        return yaml.load(fh, Loader=yaml.FullLoader)


def configure_logging(output, debug):
    with resource_stream(__name__, 'logging.yml') as stream:
        loggingConfig = yaml.load(stream, Loader=yaml.FullLoader)

    for k,v in loggingConfig["handlers"].items():
        if k == "file":
            v["filename"] = str(output / v["filename"])
        if (k == "console") and debug:
            v["level"] = "DEBUG"

    logging.config.dictConfig(loggingConfig)
