from scip.data_masking import mask_creation, mask_apply
from scip.utils import util
from scip.data_normalization import quantile_normalization
from scip.quality_control import intensity_distribution
from scip.data_features import feature_extraction, cellprofiler
import time
import click
import logging
import logging.config
from pathlib import Path
import dask.bag
import matplotlib.pyplot as plt
import shutil
from functools import partial
from importlib import import_module
import yaml
from pkg_resources import resource_stream


def main(*, paths, output_directory, n_workers, headless, debug, port, local, config):

    with resource_stream(__name__, 'logging.yml') as stream:
        loggingConfig = yaml.load(stream, Loader=yaml.FullLoader)
    logging.config.dictConfig(loggingConfig)

    logger = logging.getLogger("scip")
    logger.info(f"Running pipeline for {','.join(paths)}")

    # logic for creating output directory
    should_remove = True
    output_dir = Path(output_directory)
    if (not headless) and output_dir.exists():
        should_remove = click.prompt(
            f"{str(output_dir)} exists. Overwrite contents? [Y/n]",
            type=str, show_default=False, default="Y"
        ) == "Y"

        if not should_remove:
            raise FileExistsError(f"{str(output_dir)} exists and should not be removed. Exiting.")
    if should_remove and output_dir.exists():
        logger.info(f"Running headless and/or {str(output_dir)} exists. Removing.")
        shutil.rmtree(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    config = util.load_yaml_config(config)

    start_full = time.time()

    # ClientClusterContext creates cluster
    # and registers Client as default client for this session
    logger.debug("Starting Dask cluster")
    with util.ClientClusterContext(n_workers=n_workers, local=local, port=port) as context:
        logger.debug(f"Client ({context}) created")

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

        report_made = intensity_distribution.segmentation_intensity_report(
            images, 100, channel_amount, output_dir)
        images = intensity_distribution.check_report(images, report_made)

        features = feature_extraction.extract_features(images)
        cp_features = cellprofiler.extract_features(images=images, channels=channels)

        if debug:
            features.visualize(filename=str(output_dir / "task_graph.svg"))

        # some images are exported for demonstration purposes
        fig, grid = plt.subplots(5, 4)
        for im, axes in zip(images.take(5), grid):
            axes[0].set_title(im["path"])
            axes[0].imshow(im["pixels"][1])
            axes[1].imshow(im["denoised"][1])
            axes[2].imshow(im["segmented"][1])
            axes[3].imshow(im["mask"][1])
        plt.savefig(output_dir / "output_images.png")

        features.compute().to_parquet(str(output_dir / "features.parquet"))
        cp_features.compute().to_parquet(str(output_dir / "cp_features.parquet"))

        if debug:
            context.client.profile(filename=output_dir / "profile.html")

    logger.info(f"Full runtime {(time.time() - start_full):.2f}")


@click.command(name="Scalable imaging pipeline")
@click.argument("paths", nargs=-1, type=click.Path(exists=True, file_okay=False))
@click.argument("output_directory", type=click.Path(file_okay=False), default="tmp")
@click.argument(
    "config", type=click.Path(dir_okay=False, exists=True), help="Path to YAML config file")
@click.option(
    "--n-workers", "-j", type=int, default=-1,
    help="how many workers are started in the dask cluster")
@click.option("--port", "-p", type=int, default=8787, help="dask dashboard port")
@click.option("--debug", envvar="DEBUG", is_flag=True, help="sets logging level to debug")
@click.option(
    "--local/--no-local", default=True,
    help="deploy app to Dask LocalCluster, otherwise deploy to dask-jobqueue PBSCluster")
@click.option(
    "--headless", default=False, is_flag=True,
    help="If set, the program will never ask for user input")
def cli(**kwargs):
    """Intro documentation
    """
    main(**kwargs)


if __name__ == "__main__":
    import os

    # add DEBUG_DATASET entry to terminal.integrated.env.linux in VS Code workspace settings
    # should contain path to small debug dataset
    path = os.environ["DEBUG_DATASET"]
    main(
        paths=(path,),
        output_directory="tmp",
        headless=True,
        config='/home/maximl/daskPipeline/scip.yml',
        debug=True, n_workers=2, port=9003, local=True)
