import dask
from data_loading import multiframe_tiff
from data_masking import mask_creation
from utils import util

def main():

    # get_client creates local cluster
    # and registers Client as default client for this session
    util.get_client(local=True)

    path = "/group/irc/shared/vulcan_pbmc_debug"
    images = multiframe_tiff.from_directory(path)
    images = mask_creation.create_masks(images)
    images = dask.compute(*images)

if __name__ == "__main__":
    main()