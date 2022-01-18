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

import time
import os
import socket
import logging
import logging.config
from datetime import datetime
from typing import List, Tuple
from pathlib import Path
from functools import partial
from importlib import import_module

import click
import dask.bag
import dask.dataframe
import pandas

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


def get_images_bag(
    *,
    paths: List[str],
    channels: List[int],
    config: dict,
    partition_size: int,
    gpu_accelerated: bool,
    limit: int,
    loader_module
) -> Tuple[dask.bag.Bag, int, dict]:

    loader = partial(
        loader_module.bag_from_directory,
        channels=channels,
        partition_size=partition_size,
        gpu_accelerated=gpu_accelerated,
        **(config["loading"]["loader_kwargs"] or dict()))

    images = []
    maximum_pixel_value = 0

    for path in paths:

        if limit == 0:
            break

        logging.info(f"Bagging {path}")
        bag, loader_meta, maximum_pixel_value, length = loader(path=path, limit=limit)
        images.append(bag)

        limit -= length

    images = dask.bag.concat(images)
    return images, maximum_pixel_value, loader_meta


def compute_features(images, channel_names, types, maximum_pixel_value, loader_meta):

    def rename(c):
        if any(c.startswith(a) for a in ["bbox", "regions"] + list(loader_meta.keys())):
            return f"meta_{c}"
        else:
            return f"feat_{c}"

    features = feature_extraction.extract_features(
        images=images,
        channel_names=channel_names,
        types=types,
        maximum_pixel_value=maximum_pixel_value,
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
    if quantiles is not None:
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
    paths,
    output,
    config,
    partition_size,
    n_workers,
    n_nodes,
    n_cores,
    n_threads,
    memory,
    walltime,
    project,
    job_extra,
    mode,
    local_directory,
    headless,
    port,
    debug,
    timing,
    report,
    gpu,
    limit
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
            gpu=gpu
    ) as context:

        output = Path(output)
        util.make_output_dir(output, headless=headless)

        util.configure_logging(output, debug)
        logger = logging.getLogger("scip")

        version = get_versions()["version"]
        logger.info(f"SCIP version {version}")

        t = datetime.utcnow().isoformat(timespec="seconds")
        logger.info(f"Starting at {t}")
        logger.info(f"Running pipeline for {','.join(paths)}")
        logger.info(f"Running with {n_workers} workers and {n_threads} threads per worker")
        logger.info(f"Mode: {mode}")
        logger.info(f"GPUs: {gpu}")
        logger.info(f"Partition size: {partition_size}")
        logger.info(f"Exporting reports? {report}")
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
            ]])
        logger.info(f"Running with following config: {config}")

        host = context.client.run_on_scheduler(socket.gethostname)
        port = context.client.scheduler_info()['services']['dashboard']
        logger.info(f"Dashboard -> ssh -N -L {port}:{host}:{port}")

        template_dir = os.path.dirname(__file__) + "/reports/templates"

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
            images, maximum_pixel_value, loader_meta = get_images_bag(
                paths=paths,
                channels=channels,
                config=config,
                partition_size=partition_size,
                gpu_accelerated=gpu > 0,
                limit=limit,
                loader_module=loader_module
            )

        futures = []
        if report:
            import matplotlib  # noqa: E402
            matplotlib.use("Agg")
            from scip.reports import example_images, intensity_distribution  # noqa: E402
            logger.debug("reporting example images")
            futures.append(example_images.report(
                images,
                template_dir=template_dir,
                template="example_images.html",
                name="raw",
                output=output
            ))
            logger.debug("reporting on image distributions")
            futures.append(intensity_distribution.report(
                images,
                template_dir=template_dir,
                template="intensity_distribution.html",
                bin_amount=20,
                channel_names=channel_names,
                output=output,
                name="raw"
            ))

        method = config["masking"]["method"]
        if method is not None:
            masking_module = import_module('scip.masking.%s' % config["masking"]["method"])
            logger.debug("creating masks on bag")

            # in the main phase of the masking procedure, only the main
            # channel is masked and that mask is applied to all other channels
            images = masking_module.create_masks_on_bag(
                images,
                main=True,
                main_channel=config["masking"]["bbox_channel_index"],
                **(config["masking"]["kwargs"] or dict())
            )

            if report:
                from scip.reports import masks  # noqa: E402
                logger.debug("mask report")
                futures.append(masks.report(
                    images,
                    template_dir=template_dir,
                    template="masks.html",
                    name="masked",
                    output=output,
                    channel_names=channel_names
                ))

            logger.debug("preparing bag for feature extraction")

            func = partial(
                masking_util.mask_predicate,
                bbox_channel_index=config["masking"]["bbox_channel_index"]
            )
            images = images.map(func)

            # in the non-main phase of the masking procedure, the masks for the non-main
            # channels are computed and applied
            images = masking_module.create_masks_on_bag(
                images,
                main=False, main_channel=config["masking"]["bbox_channel_index"],
                **(config["masking"]["kwargs"] or dict())
            )

            images = images.map_partitions(
                masking_util.remove_regions_touching_border_partition,
                bbox_channel_index=config["masking"]["bbox_channel_index"]
            )

            images = images.map_partitions(masking_util.bounding_box_partition)

            if config["masking"]["export"]:
                no_pixels = images.map(remove_pixels)
                no_pixels.to_avro(
                    filename=str(output / "masks.*.avro"),
                    schema={
                        "name": "events",
                        "type": "record",
                        "fields": get_schema(no_pixels.take(1))
                    }
                )

            # mask is applied and background values are computed
            images = images.map_partitions(masking_util.apply_mask_partition)

            if report:
                logger.debug("reporting example images")
                futures.append(example_images.report(
                    images.filter(lambda p: "pixels" in p),
                    template_dir=template_dir,
                    template="example_images.html",
                    name="masked",
                    output=output
                ))

        if config["filter"] is not None:
            filter_module = import_module('scip.loading.%s' % config["filter"]["name"])

            images = images.map_partitions(filter_module.feature_partition)
            images = images.map(copy_without, without=["pixels"]).persist()
            filter_items = filter_module.grouped_item(images, key="group")

            images = images.map(filter_module.predicate, **filter_items)

            images = images.map_partitions(
                loader_module.reload_image_partition,
                channels=channels,
                **(config["loading"]["loader_kwargs"] or dict())
            )

        quantiles = None
        if config["normalization"] is not None:
            logger.debug("performing normalization")
            from scip.normalization import quantile_normalization  # noqa: E402
            images, quantiles = quantile_normalization.quantile_normalization(
                images,
                config["normalization"]["lower"],
                config["normalization"]["upper"],
                len(channels)
            )
            futures.append(channel_boundaries(quantiles, config=config, output=output))
            if report:
                filtered_images = images.filter(lambda p: "pixels" in p)
                logger.debug("reporting example masked images")
                futures.append(example_images.report(
                    filtered_images,
                    template_dir=template_dir,
                    template="example_images.html",
                    name="normalized",
                    output=output
                ))

                logger.debug("reporting distribution of masked images")
                futures.append(intensity_distribution.report(
                    filtered_images,
                    template_dir=template_dir,
                    template="intensity_distribution.html",
                    bin_amount=20,
                    channel_names=channel_names,
                    output=output,
                    name="normalized"
                ))

        logger.debug("computing features")
        bag_df = compute_features(
            images=images,
            channel_names=channel_names,
            types=config["feature_extraction"]["types"],
            maximum_pixel_value=maximum_pixel_value,
            loader_meta=loader_meta
        )
        bag_df = bag_df.repartition(npartitions=5)

        filename = config["export"]["filename"]
        futures.append(
            bag_df.to_parquet(
                str(output),
                name_function=lambda x: f"{filename}.{x}.parquet",
                write_metadata_file=False,
                engine="fastparquet"
            )
        )

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


@click.command(name="Scalable imaging pipeline")
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
    "--limit", default=-1, type=int,
    help="Limit the number of events to load from each file or directory"
)
@click.option("--timing", default=None, type=click.Path(dir_okay=False))
@click.option("--report/--no-report", default=True, is_flag=True, type=bool)
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
@click.argument("paths", nargs=-1, type=click.Path(exists=True))
def cli(**kwargs):
    """Intro documentation
    """

    # noop if no paths are provided
    if len(kwargs["paths"]) == 0:
        return

    runtime = main(**kwargs)

    timing = kwargs["timing"]
    if timing is not None:
        import json
        with open(timing, "w") as fp:
            json.dump({**kwargs, **dict(runtime=runtime)}, fp)
        logging.getLogger("scip").info(f"Timing output written to {timing}")


if __name__ == "__main__":
    cli()
