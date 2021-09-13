import matplotlib
matplotlib.use("Agg")

from scip.utils import util
from scip.normalization import quantile_normalization
from scip.reports import feature_statistics, example_images, intensity_distribution, masks
from scip.features import feature_extraction, cellprofiler
from scip.segmentation.mask_apply import get_masked_intensities
# from scip.analysis import fuzzy_c_mean
import time
import click
import logging
import logging.config
from pathlib import Path
import dask.bag
import dask.dataframe
import shutil
from functools import partial
from importlib import import_module
import numpy
import os


def flat_intensities_partition(part):

    def get_flat_intensities(p):
        out = p.copy()
        out["flat"] = p["pixels"].reshape(p["pixels"].shape[0], -1)
        return out

    return [get_flat_intensities(p) for p in part]


def masked_intensities_partition(part):
    return [get_masked_intensities(p) for p in part]


def get_images_bag(paths, channels, config, partition_size):

    loader_module = import_module('scip.loading.%s' % config["loading"]["format"])
    loader = partial(
        loader_module.bag_from_directory,
        channels=channels,
        partition_size=partition_size)

    images = []
    idx = 0
    for path in paths:
        assert Path(path).exists(), f"{path} does not exist."
        assert Path(path).is_dir(), f"{path} is not a directory."
        logging.info(f"Bagging {path}")
        bag, idx = loader(path, idx)
        images.append(bag)

    return dask.bag.concat(images)


def preprocess_bag(bag):

    # images are loaded from directory and masked
    # after this operation the bag is persisted as it
    # will be reused several times throughout the pipeline
    bag = bag.map_partitions(masked_intensities_partition)
    bag = quantile_normalization.quantile_normalization(bag, 0, 1)

    return bag


def compute_features(images, channels, prefix):

    skimage_features = feature_extraction.extract_features(images=images)
    cp_features = cellprofiler.extract_features(images=images, channels=channels)
    features = dask.dataframe.multi.concat([skimage_features, cp_features], axis=1)

    def name(c):
        parts = c.split("_", 1)
        return f"{parts[0]}_{prefix}_{parts[1]}"
    features = features.rename(columns=name)

    return features


def main(
    *,
    paths,
    output,
    config,
    partition_size,
    n_workers,
    n_processes,
    n_cores,
    memory,
    walltime,
    job_extra,
    local,
    local_directory,
    headless,
    port,
    debug,
    timing
):

    # logic for creating output directory
    should_remove = True

    output = Path(output)
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

    util.configure_logging(output, debug)
    logger = logging.getLogger("scip")
    logger.info(f"Running pipeline for {','.join(paths)}")
    logger.info(f"Running with {n_workers} workers/nodes and {n_processes} processes")
    logger.info(f"Local mode? {local}")
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
            n_processes=n_processes,
            local_directory=local_directory,
            cores=n_cores,
            memory=memory,
            walltime=walltime,
            job_extra=job_extra
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
        channel_labels = [f'ch{i}' for i in channels]

        images = get_images_bag(paths, channels, config, partition_size)
        logger.debug("Loaded images")
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
        for k, v in bags.items():
            masks.report(v, channel_labels=channel_labels, output=output, name=k)

        # with open("test/data/masked.pickle", "wb") as fh:
        #     import pickle
        #     pickle.dump(bags["otsu"].compute(), fh)
        # return

        def nonempty_mask_predicate(s):
            flat = s["mask"].reshape(s["mask"].shape[0], -1)
            return all(numpy.any(flat, axis=1))
        
        feature_dataframes = []
        for k, bag in bags.items():
            
            bag = bag.filter(nonempty_mask_predicate)
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

            feature_dataframes.append(compute_features(bag, channels, k))

        features = dask.dataframe.multi.concat(feature_dataframes, axis=1)
        features = features.persist()

        # memberships, membership_plot = fuzzy_c_mean.fuzzy_c_means(features, 5, 3, 10)
        # if output is not None:
            # membership_plot.compute()

        filename = config["export"]["filename"]
        features.compute().to_parquet(str(output / f"{filename}.parquet"))
        feature_statistics.report(
            features, 
            template_dir=template_dir, 
            template="feature_statistics.html",
            output=output
        )

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
    help="Number of workers in the LocalCluster, or number of provisioned nodes otherwise")
@click.option(
    "--n-processes", "-n", type=int, default=1,
    help="Number of workers started per node in the PBSCluster")
@click.option(
    "--n-cores", "-c", type=click.IntRange(min=1), default=1,
    help="Number of cores available per node in the cluster")
@click.option(
    "--memory", "-m", type=click.IntRange(min=1), default=4,
    help="Amount of memory available per node in the cluster")
@click.option(
    "--walltime", "-w", type=str, default="01:00:00",
    help="Expected required walltime for the job to finish")
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
    import os

    # add DEBUG_DATASET entry to terminal.integrated.env.linux in VS Code workspace settings
    # should contain path to small debug dataset
    path = os.environ["FULL_DATASET"]
    main(
        paths=(path,),
        output="tmp",
        headless=True,
        config='scip.yml',
        partition_size=50,
        debug=True, n_workers=2, n_processes=4, port=8787, local=True)
