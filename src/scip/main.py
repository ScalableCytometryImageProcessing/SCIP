from scip.data_masking import mask_creation
from scip.utils import util
from scip.data_normalization import quantile_normalization
from scip.quality_control import feature_statistics, visual, intensity_distribution
from scip.data_features import feature_extraction, cellprofiler
from scip.data_masking.mask_apply import get_masked_intensities
# from scip.data_analysis import fuzzy_c_mean
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


def flat_intensities_partition(part):

    def get_flat_intensities(p):
        out = p.copy()
        out["flat"] = p["pixels"].reshape(p["pixels"].shape[0], -1)
        return out

    return [get_flat_intensities(p) for p in part]


def masked_intensities_partition(part):
    return [get_masked_intensities(p) for p in part]


def get_images_bag(paths, channels, config):

    loader_module = import_module('scip.data_loading.%s' % config["data_loading"]["format"])
    loader = partial(
        loader_module.bag_from_directory,
        channels=channels,
        partition_size=config["data_loading"]["partition_size"])

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
    bag = quantile_normalization.quantile_normalization(bag, 0.01, 0.99)

    return bag


def compute_features(images, channels, prefix):

    skimage_features = feature_extraction.extract_features(images=images)
    cp_features = cellprofiler.extract_features(images=images, channels=channels)
    features = dask.dataframe.multi.concat([skimage_features, cp_features], axis=1)
    features = features.rename(columns=lambda c: prefix + "_" + c)

    return features


def main(*, paths, output, n_workers, headless, debug, n_processes, port, local, config):

    util.configure_logging()
    logger = logging.getLogger("scip")
    logger.info(f"Running pipeline for {','.join(paths)}")

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
        logger.info(f"Running headless and/or {str(output)} exists. Removing.")
        shutil.rmtree(output)
    if not output.exists():
        output.mkdir(parents=True)

    config = util.load_yaml_config(config)

    # ClientClusterContext creates cluster
    # and registers Client as default client for this session
    logger.debug("Starting Dask cluster")
    with util.ClientClusterContext(n_workers=n_workers, local=local,
                                   port=port, n_processes=n_processes) as context:
        logger.debug(f"Client ({context}) created")

        start = time.time()

        assert "channels" in config["data_loading"], "Please specify what channels to load"
        channels = config["data_loading"].get("channels")

        images = get_images_bag(paths, channels, config)
        bags = mask_creation.create_masks_on_bag(images, noisy_channels=[0])

        # with open("test/data/masked.pickle", "wb") as fh:
        #     import pickle
        #     pickle.dump(bags["otsu"].compute(), fh)
        # return

        images = images.map_partitions(flat_intensities_partition)
        intensity_distribution.segmentation_intensity_report(
            bag=images,
            bin_amount=100,
            channels=channels,
            output=output,
            name="raw"
        ).compute()

        features = []
        for k, bag in bags.items():
            bag = preprocess_bag(bag)
            bag = bag.persist()

            visual.plot_images(bag, title=k, output=output)
            intensity_distribution.segmentation_intensity_report(
                bag=bag,
                bin_amount=100,
                channels=channels,
                output=output,
                name=k,
                extent=numpy.array([(0, 1)] * len(channels)) # extent is known due to normalization
            ).compute()

            # features.append(compute_features(bag, channels, k))
        # features = dask.dataframe.multi.concat(features, axis=1)
        # features = features.persist()

        # memberships, membership_plot = fuzzy_c_mean.fuzzy_c_means(features, 5, 3, 10)
        # if output is not None:
        #     membership_plot.compute()

        # filename = config["data_export"]["filename"]
        # features.compute().to_parquet(str(output / f"{filename}.parquet"))
        # feature_statistics.get_feature_statistics(features, output).compute()

        # if debug:
        #     context.client.profile(filename=str(output / "profile.html"))

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
    "--headless", default=False, is_flag=True,
    help="If set, the program will never ask for user input")
@click.option("--timing", default=None, type=click.Path(dir_okay=False))
@click.option("--output", "-o", default=None, type=click.Path(file_okay=False))
@click.argument("config", type=click.Path(dir_okay=False, exists=True))
@click.argument("paths", nargs=-1, type=click.Path(exists=True, file_okay=False))
def cli(**kwargs):
    """Intro documentation
    """

    # noop if no paths are provided
    if len(kwargs["paths"]) == 0:
        return

    timing = None
    if kwargs["timing"] is not None:
        timing = kwargs["timing"]
    del kwargs["timing"]

    runtime = main(**kwargs)

    if timing is not None:
        import json
        with open(timing, "w") as fp:
            json.dump({**kwargs, **dict(runtime=runtime)}, fp)
        logging.getLogger("scip").info(f"Timing output written to {timing}.json")


if __name__ == "__main__":
    import os

    # add DEBUG_DATASET entry to terminal.integrated.env.linux in VS Code workspace settings
    # should contain path to small debug dataset
    path = os.environ["DEBUG_DATASET"]
    main(
        paths=(path,),
        output="tmp",
        headless=True,
        config='scip.yml',
        debug=True, n_workers=4, n_processes=1, port=8787, local=True)
