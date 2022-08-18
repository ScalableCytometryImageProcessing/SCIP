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

from typing import Any, Optional, List

import time
import os
import socket
import logging
import logging.config
from datetime import datetime
from pathlib import Path
from importlib import import_module

import click
import dask.bag
import dask.dataframe
import dask.dataframe.multi
import pandas

from scip.loading.util import get_images_bag
from scip.utils.util import copy_without
from scip.utils import util  # noqa: E402
from scip.features import feature_extraction  # noqa: E402
from scip.masking import util as masking_util
from scip._version import get_versions

# dask issues a warning during normalization
# when initializing the map-reduce operation
# this warning can only be fixed by fixing dask
import warnings
warnings.simplefilter("ignore", category=FutureWarning)


def compute_features(images, channel_names, types, loader_meta, prefix):

    def rename(c):
        if any(c.startswith(a) for a in list(loader_meta.keys())):
            return f"meta_{c}"
        elif any(c.startswith(a) for a in ["bbox", "regions"]):
            return f"meta_{prefix}_{c}"
        else:
            if prefix is not None:
                return f"feat_{prefix}_{c}"
            else:
                return f"feat_{c}"

    features = feature_extraction.extract_features(
        images=images,
        channel_names=channel_names,
        types=types,
        loader_meta=loader_meta
    )
    features = features.rename(columns=rename)
    return features


def get_schema(event):
    py_to_avro = {
        "str": "string",
        "int": "int",
        "list": {
            "type": "array",
            "name": "mask",
            "items": "int"
        }
    }
    tmp = event[0]
    tmp["mask"] = tmp["mask"].tolist()
    tmp["bbox"] = list(tmp["bbox"])
    return [
        {"name": k, "type": py_to_avro[type(v).__name__]}
        for k, v in tmp.items()
    ]


def remove_pixels(event):
    newevent = copy_without(event, ["pixels", "mask"])
    newevent["shape"] = list(event["mask"].shape)
    newevent["mask"] = event["mask"].ravel()
    return newevent


@dask.delayed
def channel_boundaries(quantiles, *, config, output):
    data = []
    index = []
    for k, v in quantiles:
        index.append(k)
        out = {}
        for channel, r in zip(config["loading"]["channel_names"], v):
            out[f"{channel}_min"] = r[0]
            out[f"{channel}_max"] = r[1]
        data.append(out)
    pandas.DataFrame(data=data, index=index).to_csv(str(output / "channel_boundaries.csv"))


def main(  # noqa: C901
    *,
    paths: List[str],
    output: Path,
    config: str,
    mode: str,
    limit: Optional[int] = -1,
    with_replacement: Optional[bool] = False,
    partition_size: Optional[int] = 1,
    n_workers: Optional[int] = 1,
    n_nodes: Optional[int] = 1,
    n_cores: Optional[int] = None,
    n_threads: Optional[int] = 1,
    memory: Optional[int] = 1,
    walltime: Optional[str] = None,
    project: Optional[str] = "",
    job_extra: Optional[str] = "",
    local_directory: Optional[str] = "tmp",
    headless: Optional[bool] = False,
    port: Optional[int] = 8787,
    debug: Optional[bool] = False,
    timing: Optional[Any] = None,
    gpu: Optional[int] = 0,
    scheduler_adress: Optional[str] = None
):
    with util.ClientClusterContext(
            n_workers=n_workers,
            mode=mode,
            port=port,
            n_nodes=n_nodes,
            local_directory=local_directory,
            cores=n_cores,
            memory=memory,
            walltime=walltime,
            job_extra=job_extra,
            threads_per_process=n_threads,
            project=project,
            gpu=gpu,
            scheduler_adress=scheduler_adress
    ) as context:

        if not hasattr(context, "client"):
            return

        output = Path(output)
        util.make_output_dir(output, headless=headless)

        util.configure_logging(output, debug)
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
        logger.info(f"Partition size: {partition_size}")
        logger.info(f"Output is saved in {str(output)}")

        config = util.load_yaml_config(config)
        assert all([
            k in config
            for k in [
                "filter",
                "normalization",
                "loading",
                "masking",
                "feature_extraction",
                "export"
            ]]), "Config is incomplete."
        logger.info(f"Running with following config: {config}")

        host = context.client.run_on_scheduler(socket.gethostname)
        port = context.client.scheduler_info()['services']['dashboard']
        logger.info(f"Dashboard -> ssh -N -L {port}:{host}:{port}")

        # if timing is set, wait for the cluster to be fully ready
        # to isolate cluster startup time from pipeline execution
        if timing is not None:
            logger.debug("waiting for all workers")
            context.wait()

        logger.debug("timer started")
        start = time.time()

        assert "channels" in config["loading"], "Please specify what channels to load"
        channels = config["loading"]["channels"]
        channel_names = config["loading"]["channel_names"]
        assert len(channels) == len(channel_names), "Please specify a name for each channel"

        logger.debug("loading images in to bags")

        loader_module = import_module('scip.loading.%s' % config["loading"]["format"])
        with dask.config.set(**{'array.slicing.split_large_chunks': False}):
            images, loader_meta = get_images_bag(
                paths=paths,
                output=output,
                channels=channels,
                config=config,
                partition_size=partition_size,
                gpu_accelerated=gpu > 0,
                loader_module=loader_module
            )

        if limit > 0:
            from dask.bag.random import sample, choices
            if with_replacement:
                images = choices(images, k=limit)
            else:
                images = sample(images, k=limit)

        futures = []
        dataframes = []

        methods = config["masking"]["methods"]
        if methods is not None:
            for method in methods:
                masking_module = import_module('scip.masking.%s' % method["method"])
                logger.debug("creating masks on bag")

                tmp_images = masking_module.create_masks_on_bag(
                    images,
                    main_channel=method["bbox_channel_index"],
                    **(method["kwargs"] or dict())
                )

                logger.debug("preparing bag for feature extraction")

                tmp_images = tmp_images.map_partitions(
                    masking_util.remove_regions_touching_border_partition,
                    bbox_channel_index=method["bbox_channel_index"]
                )

                tmp_images = tmp_images.map_partitions(masking_util.bounding_box_partition)

                if method["export"]:
                    no_pixels = tmp_images.map(remove_pixels)
                    no_pixels.to_avro(
                        filename=str(output / "masks.*.avro"),
                        schema={
                            "name": "events",
                            "type": "record",
                            "fields": get_schema(no_pixels.take(1))
                        }
                    )

                # mask is applied and background values are computed
                tmp_images = tmp_images.map_partitions(
                    masking_util.apply_mask_partition,
                    combined_indices=config["masking"]["combined_indices"]
                )

                if config["filter"] is not None:
                    filter_module = import_module('scip.filter.%s' % config["filter"]["name"])

                    tmp_images = tmp_images.map_partitions(filter_module.feature_partition)
                    tmp_images = tmp_images.map(copy_without, without=["pixels"]).persist()
                    filter_items = filter_module.item(tmp_images)

                    tmp_images = tmp_images.map(filter_module.predicate, **filter_items)

                    tmp_images = tmp_images.map_partitions(
                        loader_module.reload_image_partition,
                        channels=channels,
                        **(config["loading"]["loader_kwargs"] or dict())
                    )

                quantiles = None
                if config["normalization"] is not None:
                    logger.debug("performing normalization")
                    from scip.normalization import quantile_normalization  # noqa: E402
                    tmp_images, quantiles = quantile_normalization.quantile_normalization(
                        tmp_images,
                        config["normalization"]["lower"],
                        config["normalization"]["upper"],
                        len(channels)
                    )
                    futures.append(channel_boundaries(quantiles, config=config, output=output))

                logger.debug("computing features")
                bag_df = compute_features(
                    images=tmp_images,
                    channel_names=channel_names,
                    types=config["feature_extraction"][method["name"]],
                    loader_meta=loader_meta,
                    prefix=method["name"]
                )

                dataframes.append(bag_df)

                # set loader meta to empty dict so that meta keys are only added once
                # (for the first masking)
                loader_meta = {k: v for k, v in loader_meta.items() if "regions" in k}

        # partitions never change between masks so we can ignore unknown divisions
        bag_df = dask.dataframe.multi.concat(dataframes, axis=1, ignore_unknown_divisions=True)
        bag_df = bag_df.repartition(npartitions=10)

        filename = config["export"]["filename"]
        export_module = import_module('scip.export.%s' % config["export"]["format"])
        futures.append(export_module.export(df=bag_df, filename=filename, output=output))

        dask.compute(*futures, traverse=False, optimize_graph=False)

        if debug:
            context.client.profile(filename=str(output / "profile.html"))

    runtime = time.time() - start
    logger.info(f"Full runtime {runtime:.2f}")
    return runtime


def _print_version(ctx, param, value):
    if not value or ctx.resilient_parsing:
        return
    click.echo(get_versions()['version'])
    ctx.exit()


@click.command(
    name="Scalable imaging pipeline",
    context_settings=dict(show_default=True)
)
@click.option("--port", "-d", type=int, default=None, help="dask dashboard port")
@click.option("--debug", envvar="DEBUG", is_flag=True, help="sets logging level to debug")
@click.option(
    "--mode", default="local", type=click.Choice(util.MODES),
    help="In which mode to run Dask")
@click.option(
    "--n-workers", "-j", type=int, default=-1,
    help="Number of workers in the LocalCluster or per node")
@click.option(
    "--n-nodes", "-n", type=int, default=1,
    help="Number of nodes started")
@click.option(
    "--n-cores", "-c", type=click.IntRange(min=1), default=1,
    help="Number of cores available per node in the cluster")
@click.option(
    "--n-threads", "-t", type=click.IntRange(min=1), default=None,
    help="Number of threads per worker process")
@click.option(
    "--memory", "-m", type=click.IntRange(min=1), default=4,
    help="Amount of memory available per node in the cluster")
@click.option(
    "--limit", "-i", type=click.IntRange(min=-1), default=-1,
    help="Amount of images to sample randomly from the dataset. -1 means no sampling.")
@click.option(
    "--with-replacement", type=bool, is_flag=True, default=False,
    help="Enable sampling with replacement. Has no effect is limit is set to default.")
@click.option(
    "--walltime", "-w", type=str, default="01:00:00",
    help="Expected required walltime for the job to finish")
@click.option(
    "--project", "-p", type=str, default=None,
    help="Project name for HPC cluster")
@click.option(
    "--job-extra", "-e", type=str, multiple=True, default=[],
    help="Extra arguments for job submission")
@click.option(
    "--headless", default=False, is_flag=True,
    help="If set, the program will never ask for user input")
@click.option(
    "--partition-size", "-s", default=50, type=click.IntRange(min=1),
    help="Set partition size")
@click.option(
    "--scheduler-adress", default=None, type=str,
    help="Adress of scheduler to connect to."
)
@click.option("--timing", default=None, type=click.Path(dir_okay=False))
@click.option(
    "--gpu", default=0, type=click.IntRange(min=0), help="Specify the amount of available GPUs")
@click.option(
    "--local-directory", "-l", default=None, type=click.Path(file_okay=False, exists=True))
@click.option(
    "-V", "--version", default=False, is_flag=True, is_eager=True,
    expose_value=False, callback=_print_version,
    help="Display version information"
)
@click.argument("output", type=click.Path(file_okay=False))
@click.argument("config", type=click.Path(dir_okay=False, exists=True))
@click.argument("paths", nargs=-1, type=str)
def cli(**kwargs):
    """Intro documentation
    """

    # noop if no paths are provided
    if len(kwargs["paths"]) == 0:
        return

    def check(p: str) -> bool:
        if p.startswith("hdfs"):
            return True
        return os.path.isabs(p)
    if kwargs["mode"] == "external":
        assert check(kwargs["output"]), "Output path must be absolute in external mode."
        err = "Paths must be absolute in external mode."
        assert all([check(p) for p in kwargs["paths"]]), err

    runtime = main(**kwargs)

    if runtime is not None:
        timing = kwargs["timing"]
        if timing is not None:
            import json
            with open(timing, "w") as fp:
                json.dump({**kwargs, **dict(runtime=runtime)}, fp)
            logging.getLogger("scip").info(f"Timing output written to {timing}")


if __name__ == "__main__":
    cli()
