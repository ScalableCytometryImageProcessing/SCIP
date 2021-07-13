import dask
from data_loading import multiframe_tiff
from utils import util
import click
import logging
from pathlib import Path

def main(*, paths, debug):

    logging.basicConfig(
        level=logging.DEBUG if debug else logging.WARNING
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Running pipeline for {','.join(paths)}")

    # get_client creates local cluster
    # and registers Client as default client for this session
    client = util.get_client(local=True)
    logger.debug(f"Client ({client}) created")

    images = []
    for path in paths:
        assert Path(path).exists(), f"{path} does not exist."
        assert Path(path).is_dir(), f"{path} is not a directory."
        images.extend(multiframe_tiff.from_directory(path))

    logger.debug(f"Loading {len(images)} images")
    images = dask.compute(*images)
    logger.debug(f"{len(images)} images loaded")

@click.command(name="Scalable imaging pipeline")
@click.argument("paths", nargs=-1, type=click.Path(exists=True, file_okay=False))
@click.option("--debug", envvar="DEBUG", is_flag=True)
def cli(**kwargs):
    """Intro documentation
    """
    main(**kwargs)

if __name__ == "__main__":
    path = "/group/irc/shared/vulcan_pbmc_debug"
    
    main(path=path, debug=True)