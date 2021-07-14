from data_loading import multiframe_tiff
from data_masking import mask_creation
from utils import util
import click
import logging
from pathlib import Path
import dask


def main(*, paths, debug):

    logging.basicConfig(
        level=logging.DEBUG if debug else logging.WARNING
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Running pipeline for {','.join(paths)}")

    # ClientClusterContext creates cluster
    # and registers Client as default client for this session
    with util.ClientClusterContext() as client:
        logger.debug(f"Client ({client}) created")

        images = []
        for path in paths:
            assert Path(path).exists(), f"{path} does not exist."
            assert Path(path).is_dir(), f"{path} is not a directory."
            images.append(multiframe_tiff.bag_from_directory(path))

        images = dask.bag.concat(images)
        images = images.map(mask_creation.create_mask)
        images.compute()

        logger.debug(f"Loading {len(images)} images")
        logger.debug(f"{len(images)} images loaded")


@click.command(name="Scalable imaging pipeline")
@click.argument("paths", nargs=-1, type=click.Path(exists=True, file_okay=False))
@click.option("--debug", envvar="DEBUG", is_flag=True)
def cli(**kwargs):
    """Intro documentation
    """
    main(**kwargs)


if __name__ == "__main__":
    path = "/home/maximl/shared_scratch/images"
    main(paths=(path,), debug=True)
