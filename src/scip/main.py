import time
import click
import logging
import logging.config
from pathlib import Path
import dask.bag
import dask.dataframe
from functools import partial
from importlib import import_module
import numpy
import os
import socket
import pandas

import matplotlib
matplotlib.use("Agg")

from scip.utils import util  # noqa: E402
from scip.normalization import quantile_normalization  # noqa: E402
from scip.reports import (  # noqa: E402
    feature_statistics, example_images, intensity_distribution, masks
)  # noqa: E402
from scip.features import feature_extraction  # noqa: E402
from scip.segmentation import util as segmentation_util  # noqa: E402
# from scip.analysis import fuzzy_c_mean  # noqa: E402


def set_groupidx_partition(part, groups):
    def set_groupidx(p):
        newp = p.copy()
        newp["groupidx"] = groups.index(p["group"])
        return newp
    return [set_groupidx(p) for p in part]


def get_images_bag(paths, channels, config, partition_size):

    loader_module = import_module('scip.loading.%s' % config["loading"]["format"])
    loader = partial(
        loader_module.bag_from_directory,
        channels=channels,
        partition_size=partition_size,
        **(config["loading"]["loader_kwargs"] or dict()))

    images = []
    meta = []
    maximum_pixel_value = 0

    for i, path in enumerate(paths):
        logging.info(f"Bagging {path}")
        bag, df, maximum_pixel_value = loader(path=path, idx=i)

        images.append(bag)
        meta.append(df)

    images, meta = dask.bag.concat(images), dask.dataframe.concat(meta)

    def add_to_list(a, b):
        a.append(b["group"])
        return sorted(list(set(a)))

    def merge_lists(a, b):
        a.extend(b)
        return sorted(list(set(a)))

    groups = images.fold(binop=add_to_list, combine=merge_lists, initial=list())
    images = images.map_partitions(set_groupidx_partition, groups)

    return images, meta, groups, maximum_pixel_value


def compute_features(images, nchannels, types, maximum_pixel_value):

    def rename(c):
        if c == "idx":
            return c
        elif any([i in c for i in ["bbox", "regions"]]):
            return f"meta_{c}"
        else:
            return f"feat_{c}"

    features = feature_extraction.extract_features(
        images=images,
        nchannels=nchannels,
        types=types,
        maximum_pixel_value=maximum_pixel_value
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
    newevent = event.copy()
    del newevent["pixels"]
    newevent["shape"] = list(newevent["mask"].shape)
    newevent["mask"] = newevent["mask"].ravel()
    return newevent


@dask.delayed
def final(features, meta, reports, quantiles, groups, *, config, template_dir, output):
    if (len(reports) > 0) and (all(reports)):
        feature_statistics.report(
            features,
            template_dir=template_dir,
            template="feature_statistics.html",
            output=output
        )

    features = features.set_index("idx")
    features = pandas.concat([features, meta], axis=1)
    filename = config["export"]["filename"]
    features.to_parquet(str(output / f"{filename}.parquet"))

    if quantiles is not None:
        data = []
        index = []
        for k, v in quantiles:
            index.append(groups[k])
            out = {}
            for channel, r in zip(config["loading"]["channel_labels"], v):
                out[f"{channel}_min"] = r[0]
                out[f"{channel}_max"] = r[1]
            data.append(out)
        pandas.DataFrame(data=data, index=index).to_csv(str(output / "channel_boundaries.csv"))


def main(
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
    report
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
            project=project
    ) as context:

        output = Path(output)
        util.make_output_dir(output, headless=headless)

        util.configure_logging(output, debug)
        logger = logging.getLogger("scip")
        logger.info(f"Running pipeline for {','.join(paths)}")
        logger.info(f"Running with {n_workers} workers and {n_threads} threads per worker")
        logger.info(f"Mode? {mode}")
        logger.info(f"Output is saved in {str(output)}")

        config = util.load_yaml_config(config)
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
        channels = config["loading"].get("channels")
        channel_labels = config["loading"].get("channel_labels")
        assert len(channels) == len(channel_labels), "Please specify a label for each channel"

        logger.debug("loading images in to bags")

        with dask.config.set(**{'array.slicing.split_large_chunks': False}):
            images, meta, groups, maximum_pixel_value = get_images_bag(
                paths, channels, config, partition_size)

        reports = []
        if report:
            logger.debug("reporting example images")
            reports.append(example_images.report(
                images,
                template_dir=template_dir,
                template="example_images.html",
                name="raw",
                output=output
            ))
            logger.debug("reporting on image distributions")
            reports.append(intensity_distribution.report(
                images,
                template_dir=template_dir,
                template="intensity_distribution.html",
                bin_amount=100,
                channel_labels=channel_labels,
                output=output,
                name="raw"
            ))

        method = config["masking"]["method"]
        if method is not None:
            masking_module = import_module('scip.segmentation.%s' % config["masking"]["method"])
            logger.debug("creating masks on bag")
            images = masking_module.create_masks_on_bag(
                images,
                **(config["masking"]["kwargs"] or dict())
            )

            if report:
                logger.debug("mask report")
                reports.append(masks.report(
                    images,
                    template_dir=template_dir,
                    template="masks.html",
                    name="masked",
                    output=output,
                    channel_labels=channel_labels
                ))

            logger.debug("preparing bag for feature extraction")

            filter_func = partial(
                segmentation_util.mask_predicate,
                bbox_channel=config["masking"]["bbox_channel"]
            )
            images = images.filter(filter_func)
            images = images.map_partitions(
                segmentation_util.bounding_box_partition,
                bbox_channel=config["masking"]["bbox_channel"]
            )

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

            images = images.map_partitions(segmentation_util.crop_to_mask_partition)
            images = images.map_partitions(segmentation_util.apply_mask_partition)

        quantiles = None
        if config["normalization"] is not None:
            logger.debug("performing normalization")
            images, quantiles = quantile_normalization.quantile_normalization(
                images,
                config["normalization"]["lower"],
                config["normalization"]["upper"],
                len(channels)
            )

        logger.debug("computing features")
        bag_df = compute_features(
            images=images,
            nchannels=len(channels),
            types=config["feature_extraction"]["types"],
            maximum_pixel_value=maximum_pixel_value
        )

        if report:
            logger.debug("reporting example masked images")
            reports.append(example_images.report(
                images,
                template_dir=template_dir,
                template="example_images.html",
                name="normalized",
                output=output
            ))

            logger.debug("reporting distribution of masked images")
            tmp = numpy.array([(0, 1)] * len(channels))
            reports.append(intensity_distribution.report(
                images,
                template_dir=template_dir,
                template="intensity_distribution.html",
                bin_amount=100,
                channel_labels=channel_labels,
                output=output,
                name="normalized",
                extent=groups.apply(
                    lambda a: [(i, tmp) for i in range(len(a))])
            ))

        f = final(
            bag_df, meta, reports, quantiles, groups,
            config=config,
            output=output,
            template_dir=template_dir
        )
        f.compute()


        if debug:
            f.visualize(filename=str(output / "final.svg"))
            context.client.profile(filename=str(output / "profile.html"))

    runtime = time.time() - start
    logger.info(f"Full runtime {runtime:.2f}")
    return runtime


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
    help="Project name for HPC cluster"
)
@click.option(
    "--job-extra", "-e", type=str, multiple=True, default=[],
    help="Extra arguments for job submission")
@click.option(
    "--headless", default=False, is_flag=True,
    help="If set, the program will never ask for user input")
@click.option(
    "--partition-size", "-s", default=50, type=click.IntRange(min=1),
    help="Set partition size")
@click.option("--timing", default=None, type=click.Path(dir_okay=False))
@click.option("--report/--no-report", default=True, is_flag=True, type=bool)
@click.option(
    "--local-directory", "-l", default=None, type=click.Path(file_okay=False, exists=True))
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
