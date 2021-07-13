import dask
from data_loading import multiframe_tiff
from utils import util

def main():
    # get_client creates local cluster
    # and registers Client as default client for this session
    util.get_client(local=True)

    path = "/home/maximl/shared_scratch/images"
    images = multiframe_tiff.from_directory(path)
    images = images.compute()

if __name__ == "__main__":
    main()