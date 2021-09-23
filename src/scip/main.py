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


def flat_intensities_partition(part):

    def get_flat_intensities(p):
        out = p.copy()
        out["flat"] = p["pixels"].reshape(p["pixels"].shape[0], -1)
        return out

    return [get_flat_intensities(p) for p in part]


def get_images_bag(paths, channels, config, partition_size):

    loader_module = import_module('scip.loading.%s' % config["loading"]["format"])
    loader = partial(
        loader_module.bag_from_directory,
        channels=channels,
        partition_size=partition_size,
        **(config["loading"]["loader_kwargs"] or dict()))

    images = []
    meta = []
    idx = 0
    for path in paths:
        assert Path(path).exists(), f"{path} does not exist."
        assert Path(path).is_dir(), f"{path} is not a directory."
        logging.info(f"Bagging {path}")
        bag, df = loader(path=path, idx=idx)

        idx += len(df)
        images.append(bag)
        meta.append(df)

    return dask.bag.concat(images), dask.dataframe.concat(meta)


def preprocess_bag(bag):

    # images are loaded from directory and masked
    # after this operation the bag is persisted as it
    # will be reused several times throughout the pipeline
    bag = bag.filter(segmentation_util.nonempty_mask_predicate)
    bag = bag.map_partitions(segmentation_util.bounding_box_partition)
    bag = bag.map_partitions(segmentation_util.crop_to_mask_partition)
    bag = bag.map_partitions(segmentation_util.masked_intensities_partition)
    bag = quantile_normalization.quantile_normalization(bag, 0, 1)

    return bag


def compute_features(images, channels, prefix):

    features = feature_extraction.extract_features(images=images)
    features = features.rename(columns=lambda c: f"feat_{prefix}_{c}")

    def to_bbox_df(el):
        return dict(
            idx=el["idx"],
            bbox_minr=el["bbox"][0],
            bbox_minc=el["bbox"][1],
            bbox_maxr=el["bbox"][2],
            bbox_maxc=el["bbox"][3]
        )
    bbox = images.map(to_bbox_df)
    bbox = bbox.to_dataframe().set_index("idx")
    bbox = bbox.rename(columns=lambda c: f"meta_{prefix}_{c}")

    return dask.dataframe.multi.concat([features, bbox], axis=1)


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
    local,
    local_directory,
    headless,
    port,
    debug,
    timing
):

    output = Path(output)
    util.make_output_dir(output, headless=headless)

    util.configure_logging(output, debug)
    logger = logging.getLogger("scip")
    logger.info(f"Running pipeline for {','.join(paths)}")
    logger.info(f"Running with {n_workers} workers and {n_threads} threads per worker")
    logger.info(f"Local mode? {local}")
    if not local:
        logger.info(f"Provisining {n_nodes} nodes")
    logger.info(f"Output is saved in {str(output)}")

    config = util.load_yaml_config(config)
    logger.info(f"Running with following config: {config}")

    template_dir = os.path.dirname(__file__) + "/reports/templates"

    # ClientClusterContext creates cluster
    # and registers Client as default client for this session
    logger.debug("Starting Dask cluster")
    logger.debug(walltime)
    with util.ClientClusterContext(
            n_workers=n_workers,
            local=local,
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
        logger.debug(f"Cluster ({context.cluster}) created")
        if not local:
            logger.debug(context.cluster.job_script())

        # if timing is set, wait for the cluster to be fully ready
        # to isolate cluster startup time from pipeline execution
        if timing is not None:
            context.wait()

        start = time.time()

        assert "channels" in config["loading"], "Please specify what channels to load"
        channels = config["loading"].get("channels")
        channel_labels = config["loading"].get("channel_labels")
        assert len(channels) == len(channel_labels), "Please specify a label for each channel"

        images, meta = get_images_bag(paths, channels, config, partition_size)
        images = images.persist()

        example_images.report(
            images,
            template_dir=template_dir,
            template="example_images.html",
            name="raw",
            output=output
        )
        intensity_distribution.report(
            images.map_partitions(flat_intensities_partition),
            template_dir=template_dir,
            template="intensity_distribution.html",
            bin_amount=100,
            channel_labels=channel_labels,
            output=output,
            name="raw"
        )

        masking_module = import_module('scip.segmentation.%s' % config["masking"]["method"])
        bags = masking_module.create_masks_on_bag(images, noisy_channels=[0])

        del images

        # with open("test/data/masked.pickle", "wb") as fh:
        #     import pickle
        #     pickle.dump(bags["otsu"].compute(), fh)
        # return

        feature_dataframes = []
        for k, bag in bags.items():

            bag = bag.persist()

            masks.report(
                bag,
                template_dir=template_dir,
                template="masks.html",
                name=k,
                output=output,
                channel_labels=channel_labels
            )

            bag = preprocess_bag(bag)
            bag = bag.persist()

            example_images.report(
                bag,
                template_dir=template_dir,
                template="example_images.html",
                name=k,
                output=output
            )
            intensity_distribution.report(
                bag,
                template_dir=template_dir,
                template="intensity_distribution.html",
                bin_amount=100,
                channel_labels=channel_labels,
                output=output,
                name=k,
                extent=numpy.array([(0, 1)] * len(channels))  # extent is known due to normalization
            )

            df = compute_features(bag, channels, k)
            feature_dataframes.append(df)

        features = dask.dataframe.multi.concat(feature_dataframes, axis=1)

        # once features are computed, pull to local
        features = dask.dataframe.multi.concat(
            [features, meta], axis=1
        ).compute()

        feature_statistics.report(
            features,
            template_dir=template_dir,
            template="feature_statistics.html",
            output=output
        )

        filename = config["export"]["filename"]
        features.to_parquet(str(output / f"{filename}.parquet"))

        if debug:
            context.client.profile(filename=str(output / "profile.html"))

    runtime = time.time() - start
    logger.info(f"Full runtime {runtime:.2f}")
    return runtime


@click.command(name="Scalable imaging pipeline")
@click.option("--port", "-p", type=int, default=None, help="dask dashboard port")
@click.option("--debug", envvar="DEBUG", is_flag=True, help="sets logging level to debug")
@click.option(
    "--local/--no-local", default=True,
    help="deploy app to Dask LocalCluster, otherwise deploy to dask-jobqueue PBSCluster")
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
@click.option(
    "--local-directory", "-d", default=None, type=click.Path(file_okay=False, exists=True))
@click.argument("output", type=click.Path(file_okay=False))
@click.argument("config", type=click.Path(dir_okay=False, exists=True))
@click.argument("paths", nargs=-1, type=click.Path(exists=True, file_okay=False))
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
