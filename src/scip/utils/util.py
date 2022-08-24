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

import socket
from dask.distributed import Client
from pathlib import Path
import yaml
from pkg_resources import resource_stream
import logging
import math
import shutil
import click
from datetime import datetime, timedelta
import dask
from scip._version import get_versions

MODES = ["local", "jobqueue", "mpi", "external"]


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
            project=None,
            gpu=0,
            scheduler_adress: str = None,
    ):
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
        self.gpu = gpu
        self.scheduler_adress = scheduler_adress

    def __enter__(self):
        if self.mode == "local":
            from dask.distributed import LocalCluster
            with dask.config.set({"distributed.worker.resources.cellpose": 1}):
                self.cluster = LocalCluster(
                    n_workers=self.n_workers, threads_per_worker=self.threads_per_process,
                    processes=True
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
            from dask_mpi import initialize
            from mpi4py import MPI

            worker_options = {}
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()

            if (self.gpu > 0) and rank in range(2, 2 + self.gpu):
                worker_options["resources"] = {'cellpose': 1}

            is_client = initialize(
                dashboard=True,
                dashboard_address=None if self.port is None else f':{self.port}',
                nthreads=self.threads_per_process,
                local_directory=self.local_directory,
                memory_limit=int(self.memory * 1e9),
                worker_class='distributed.Nanny',
                worker_options=worker_options,
                exit=False
            )

            if is_client:
                self.client = Client()
                self.client.wait_for_workers(n_workers=self.n_workers)
        elif self.mode == "external":
            assert self.scheduler_adress is not None, "Adress must be set in external mode."
            self.client = Client(address=self.scheduler_adress)

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.mode == "mpi" and hasattr(self, "client"):
            from dask_mpi import send_close_signal
            send_close_signal()
        elif hasattr(self, "client"):
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


def copy_without(event, without=[]):
    return {
        k: v for k, v in event.items()
        if k not in without
    }


def check(func):
    def inner(sample, *args, **kwargs):
        if "pixels" in sample:
            return func(sample, *args, **kwargs)
        else:
            return sample
    return inner


def prerun(context, paths, output, headless, debug, mode, gpu, n_partitions, n_threads):

    make_output_dir(output, headless=headless)

    configure_logging(output, debug)
    logger = logging.getLogger("scip")

    version = get_versions()["version"]
    logger.info(f"SCIP version {version}")

    t = datetime.utcnow().isoformat(timespec="seconds")
    logger.info(f"Starting at {t}")
    logger.info(f"Running pipeline for {','.join(paths)}")

    n_workers = len(context.client.scheduler_info()["workers"])
    logger.info(f"Running with {n_workers} workers and {n_threads} threads per worker")
    logger.info(f"Mode: {mode}")
    logger.info(f"GPUs: {gpu}")
    logger.info(f"Number of partitions: {n_partitions}")
    logger.info(f"Output is saved in {str(output)}")

    host = context.client.run_on_scheduler(socket.gethostname)
    port = context.client.scheduler_info()['services']['dashboard']
    logger.info(f"Dashboard -> ssh -N -L {port}:{host}:{port}")
