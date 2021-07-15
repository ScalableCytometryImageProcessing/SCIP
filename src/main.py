from data_loading import multiframe_tiff
from data_masking import mask_creation
from quality_control import intensity_distribution
from utils import util
import time
import click
import logging
from pathlib import Path
import dask.bag
import matplotlib.pyplot as plt
import shutil


def main(*, paths, output_directory, n_workers, headless, debug, port, local):

    logging.basicConfig(
        level=logging.DEBUG if debug else logging.WARNING
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logger = logging.getLogger(__name__)
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
        logging.info(f"Running headless and/or {str(output_dir)} exists. Removing.")
        shutil.rmtree(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    start_full = time.time()

    # ClientClusterContext creates cluster
    # and registers Client as default client for this session
    with util.ClientClusterContext(n_workers=n_workers, local=local, port=port) as context:
        logger.debug(f"Client ({context}) created")

        images = []
        for path in paths:
            assert Path(path).exists(), f"{path} does not exist."
            assert Path(path).is_dir(), f"{path} is not a directory."
            logging.info(f"Bagging {path}")
            images.append(multiframe_tiff.bag_from_directory(path, partition_size=50))

        # images are loaded from directory and masked
        # after this operation the bag is persisted as it
        # will be reused several times throughout the pipeline
        images = dask.bag.concat(images)
        images = mask_creation.create_masks_on_bag(images)
        images = context.client.persist(images)

        # QC
        # channels are compared before and after masking
        intensity_count, masked_intensity_count, bins, masked_bins = \
            intensity_distribution.get_distributed_counts(images)
        intensity_distribution.plot_before_after_distribution(
            intensity_count, bins, masked_intensity_count, masked_bins, output_dir=output_dir)

        # some images are exported for demonstration purposes
        fig, grid = plt.subplots(5, 4)
        for im, axes in zip(images.take(5), grid):
            axes[0].set_title(im["path"])
            axes[0].imshow(im["pixels"][1])
            axes[1].imshow(im["denoised"][1])
            axes[2].imshow(im["segmented"][1])
            axes[3].imshow(im["mask"][1])
        plt.savefig(output_dir / "output_images.png")

        if debug:
            context.client.profile(filename=output_dir / "profile.html")

    logger.info(f"Full runtime {(time.time() - start_full):.2f}")


@click.command(name="Scalable imaging pipeline")
@click.argument("paths", nargs=-1, type=click.Path(exists=True, file_okay=False))
@click.argument("output_directory", type=click.Path(file_okay=False), default="tmp")
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
    path = os.environ["FULL_DATASET"]
    main(
        paths=(path,),
        output_dir="tmp",
        headless=False,
        debug=True, n_workers=4, port=8990, local=False)
