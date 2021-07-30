from scip.data_masking import mask_creation, mask_apply
from scip.utils import util
from scip.data_normalization import quantile_normalization
from scip.quality_control import intensity_distribution
from scip.data_features import feature_extraction
# from scip.data_analysis import fuzzy_c_mean
import time
import click
import logging
import logging.config
from pathlib import Path
import dask.bag
import shutil
from functools import partial
from importlib import import_module


def main(*, paths, output, n_workers, headless, debug, processes, port, local, config):

    util.configure_logging()
    logger = logging.getLogger("scip")
    logger.info(f"Running pipeline for {','.join(paths)}")

    # logic for creating output directory
    should_remove = True

    if output is not None:
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
                                   port=port, processes=processes) as context:
        logger.debug(f"Client ({context}) created")

        start = time.time()

        loader_module = import_module('scip.data_loading.%s' % config["data_loading"]["format"])
        loader = partial(
            loader_module.bag_from_directory,
            channels=config["data_loading"].get("channels", None),
            partition_size=50)

        images = []
        for path in paths:
            assert Path(path).exists(), f"{path} does not exist."
            assert Path(path).is_dir(), f"{path} is not a directory."
            logging.info(f"Bagging {path}")
            images.append(loader(path))

        assert "channels" in config["data_loading"], "Please specify what channels to load"
        channels = config["data_loading"].get("channels")
        channel_amount = len(channels)

        # images are loaded from directory and masked
        # after this operation the bag is persisted as it
        # will be reused several times throughout the pipeline
        images = dask.bag.concat(images)
        images = mask_creation.create_masks_on_bag(images, noisy_channels=[0])
        images = mask_apply.create_masked_images_on_bag(images)
        images = quantile_normalization.quantile_normalization(images, 0.05, 0.95)

        # intermediate persist so that extract_features can reuse
        # above computations after masking QC reports are generated
        images = images.persist()

        if output is not None:
            report_made = intensity_distribution.segmentation_intensity_report(
                images, 100, channel_amount, output)
            images = intensity_distribution.check_report(images, report_made)

        features = feature_extraction.extract_features(images)
        # plotted, features = feature_statistics.get_feature_statistics(features)
        # plotted = True
        # features = feature_statistics.check_report(features, plotted, meta=features._meta)
        # memberships, plotted = fuzzy_c_mean.fuzzy_c_means(features, 5, 3, 10)
        # plotted.compute()
        # images = cellprofiler.check_plotted(images, plotted, meta=features._meta)
        # cp_features = cellprofiler.extract_features(images=images, channels=channels)
        # cp_features.compute()

        features = features.compute()
        # cp_features = cp_features.compute()

        if output is not None:
            filename = config["data_export"]["filename"]
            features.to_parquet(str(output / f"{filename}.parquet"))
            # cp_features.to_parquet(str(output / "cp_features.parquet"))

        if debug and output is not None:
            features.visualize(filename=str(output / "task_graph.svg"))
            context.client.profile(filename=output / "profile.html")

    runtime = time.time() - start
    logger.info(f"Full runtime {runtime:.2f}")
    return runtime


@click.command(name="Scalable imaging pipeline")
@click.option(
    "--n-workers", "-j", type=int, default=-1,
    help="how many workers are started in the dask cluster")
@click.option(
    "--processes", "-n", type=int, default=12,
    help="how many processes are started for every node in the dask cluster")
@click.option("--port", "-p", type=int, default=None, help="dask dashboard port")
@click.option("--debug", envvar="DEBUG", is_flag=True, help="sets logging level to debug")
@click.option(
    "--local/--no-local", default=True,
    help="deploy app to Dask LocalCluster, otherwise deploy to dask-jobqueue PBSCluster")
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
        headless=False,
        config='/home/sanderth/dask-pipeline/scip.yml',
        debug=True, n_workers=4, processes=12, port=8990, local=True)
