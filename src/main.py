import dask
from data_loading import multiframe_tiff
from utils import util

def main():

    # get_client creates local cluster
    # and registers Client as default client for this session
    util.get_client(local=True)

    path = "/group/irc/shared/vulcan_pbmc_debug"
    images = multiframe_tiff.from_directory(path)
    images = dask.compute(*images)

if __name__ == "__main__":
    main()