# Contains methods for administrative tasks

from dask.distributed import Client
from pathlib import Path
import yaml
from pkg_resources import resource_stream
import logging
import math
import shutil
import click
from datetime import datetime, timedelta

MODES = ["local", "jobqueue", "mpi"]


class ClientClusterContext:

    def __init__(
            self,
            *,
            mode="local",
            n_workers=12,
            n_nodes=1,
            port=8787,
            local_directory=None,
            memory=None,
            cores=None,
            job_extra=[],
            walltime="01:00:00",
            threads_per_process=None,
            project=None
    ):
        """
        Sets up a cluster and client.

        local (bool): If true, sets up a LocalCluster, otherwise a PBSCluster
        n_workers (int): Defines the amount of workers the cluster will create
        """
        self.mode = mode
        self.n_workers = n_workers
        self.port = port
        self.n_nodes = n_nodes
        self.local_directory = local_directory
        self.memory = memory
        self.cores = cores
        self.job_extra = job_extra
        self.walltime = walltime
        self.threads_per_process = threads_per_process
        self.project = project

    def __enter__(self):
        if self.mode == "local":
            from dask.distributed import LocalCluster
            self.cluster = LocalCluster(
                n_workers=self.n_workers, threads_per_worker=self.threads_per_process
            )
            self.client = Client(self.cluster)
        elif self.mode == "jobqueue":
            from dask_jobqueue import PBSCluster

            assert (Path.home() / "logs").exists(), "Make sure directory\
                 'logs' exists in your home dir"

            t = datetime.strptime(self.walltime, "%H:%M:%S")
            seconds = timedelta(hours=t.hour, minutes=t.minute, seconds=t.second).total_seconds()
            extra = ["--lifetime", f"{seconds}s"]
            if self.threads_per_process is not None:
                extra = ["--nthreads", self.threads_per_process]

            mb_needed = math.ceil(self.memory * 1073.74)
            self.cluster = PBSCluster(
                cores=self.cores,
                memory=f"{self.memory}GiB",
                processes=self.n_workers,
                resource_spec=f"nodes=1:ppn={self.cores},mem={mb_needed}mb",
                project=self.project,
                local_directory=self.local_directory,
                walltime=self.walltime,
                extra=extra,
                job_extra=self.job_extra,
                scheduler_options={
                    'dashboard_address': None if self.port is None else f':{self.port}'},
                death_timeout=10 * 60
            )

            self.cluster.scale(jobs=self.n_nodes)
            self.client = Client(self.cluster)
        elif self.mode == "mpi":
            import dask_mpi.core

            dask_mpi.core.initialize(
                dashboard=True,
                dashboard_address=None if self.port is None else f':{self.port}',
                interface="ib0",
                nthreads=self.threads_per_process,
                local_directory=self.local_directory,
                memory_limit=int(self.memory * 1e9),
                nanny=True
            )
            self.client = Client()

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.client.close()

        if exc_type is not None:
            logging.getLogger(__name__).error(
                "Exception in context: %s, %s", exc_type, str(exc_value))

    def wait(self):
        if self.mode in ["local", "mpi"]:
            n_workers = self.n_workers
        elif self.mode == "jobqueue":
            n_workers = self.n_workers * self.n_nodes

        self.client.wait_for_workers(n_workers=n_workers)


def load_yaml_config(path):
    with open(path) as fh:
        return yaml.load(fh, Loader=yaml.FullLoader)


def configure_logging(output, debug):
    with resource_stream(__name__, 'logging.yml') as stream:
        loggingConfig = yaml.load(stream, Loader=yaml.FullLoader)

    for k, v in loggingConfig["handlers"].items():
        if k == "file":
            v["filename"] = str(output / v["filename"])
        if (k == "console") and debug:
            v["level"] = "DEBUG"

    logging.config.dictConfig(loggingConfig)


def make_output_dir(output, headless):
    should_remove = True

    if (not headless) and output.exists():
        should_remove = click.prompt(
            f"{str(output)} exists. Overwrite contents? [Y/n]",
            type=str, show_default=False, default="Y"
        ) == "Y"

        if not should_remove:
            raise FileExistsError(f"{str(output)} exists and should not be removed. Exiting.")
    if should_remove and output.exists():
        shutil.rmtree(output)
    if not output.exists():
        output.mkdir(parents=True)
