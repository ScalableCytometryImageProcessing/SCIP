from data_loading import multiframe_tiff
from data_masking import mask_creation
from utils import util
import time
import click
import logging
from pathlib import Path
import dask.bag
import matplotlib.pyplot as plt


def main(*, paths, n_workers, debug):

    logging.basicConfig(
        level=logging.DEBUG if debug else logging.WARNING
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logger = logging.getLogger(__name__)
    logger.info(f"Running pipeline for {','.join(paths)}")

    start_full = time.time()

    # ClientClusterContext creates cluster
    # and registers Client as default client for this session
    with util.ClientClusterContext(n_workers=n_workers) as context:
        logger.debug(f"Client ({context}) created")

        images = []
        for path in paths:
            assert Path(path).exists(), f"{path} does not exist."
            assert Path(path).is_dir(), f"{path} is not a directory."
            logging.info(f"Bagging {path}")
            images.append(multiframe_tiff.bag_from_directory(path, partition_size=50))

        images = dask.bag.concat(images)
        images = mask_creation.create_masks_on_bag(images)

        start = time.time()
        images = images.take(5)
        logger.info(f"Compute runtime {(time.time() - start):.2f}")

        fig, grid = plt.subplots(5, 4)
        for im, axes in zip(images, grid):
            axes[0].imshow(im["pixels"][1])
            axes[1].imshow(im["denoised"][1])
            axes[2].imshow(im["segmented"][1])
            axes[3].imshow(im["mask"][1])
        plt.savefig("output_images.png")

        if debug:
            context.client.profile(filename="profile.html")

    logger.info(f"Full runtime {(time.time() - start_full):.2f}")


@click.command(name="Scalable imaging pipeline")
@click.argument("paths", nargs=-1, type=click.Path(exists=True, file_okay=False))
@click.option("--n-workers", "-j", type=int, default=-1)
@click.option("--debug", envvar="DEBUG", is_flag=True)
def cli(**kwargs):
    """Intro documentation
    """
    main(**kwargs)


if __name__ == "__main__":
    path = "/home/maximl/shared_scratch/vulcan_pbmc_debug"
    main(paths=(path,), n_workers=2, debug=True)
