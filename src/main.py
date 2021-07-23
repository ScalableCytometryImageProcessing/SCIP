from data_loading import multiframe_tiff
from data_masking import mask_creation, mask_apply
from data_features import feature_extraction
from quality_control import intensity_distribution
from data_normalization import quantile_normalization
from utils import util
import time
import click
import logging
from pathlib import Path
import dask.bag
import matplotlib.pyplot as plt


def main(*, paths, n_workers, debug, port, local):

    logging.basicConfig(
        level=logging.DEBUG if debug else logging.WARNING
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logger = logging.getLogger(__name__)
    logger.info(f"Running pipeline for {','.join(paths)}")

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

        images = dask.bag.concat(images)
        images = mask_creation.create_masks_on_bag(images, noisy_channels=[0])
        images = mask_apply.create_masked_images_on_bag(images)
        images = quantile_normalization.quantile_normalization(images, 0.05, 0.95)
        report_made = intensity_distribution.segmentation_intensity_report(images, 100)
        images = intensity_distribution.check_report(images, report_made)
        features = feature_extraction.extract_features(images)
        print(features.shape)
        start = time.time()

        logger.info(f"Compute runtime {(time.time() - start):.2f}")

        fig, grid = plt.subplots(5, 4)
        channel = 1
        for im, axes in zip(images, grid):
            axes[0].imshow(im["pixels"][channel])
            axes[1].imshow(im["denoised"][channel])
            axes[2].imshow(im["segmented"][channel])
            axes[3].imshow(im["mask"][channel])
        plt.savefig(f"output_images_ch{channel}.png")

        if debug:
            context.client.profile(filename="profile.html")

    logger.info(f"Full runtime {(time.time() - start_full):.2f}")


@click.command(name="Scalable imaging pipeline")
@click.argument("paths", nargs=-1, type=click.Path(exists=True, file_okay=False))
@click.option(
    "--n-workers", "-j", type=int, default=-1,
    help="how many workers are started in the dask cluster")
@click.option("--port", "-p", type=int, default=8787, help="dask dashboard port")
@click.option("--debug", envvar="DEBUG", is_flag=True, help="sets logging level to debug")
@click.option(
    "--local", "-l", is_flag=True, default=True,
    help="if true, deploy app to Dask LocalCluster, otherwise deploy to dask-jobqueue PBSCluster")
def cli(**kwargs):
    """Intro documentation
    """
    main(**kwargs)


if __name__ == "__main__":
    import os

    # add DEBUG_DATASET entry to terminal.integrated.env.linux in VS Code workspace settings
    # should contain path to small debug dataset
    path = os.environ["DEBUG_DATASET"]
    path = "/home/sanderth/debug_images"
    main(paths=(path,), debug=True, n_workers=2, port=8990, local=False)
