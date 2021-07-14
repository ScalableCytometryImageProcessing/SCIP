from data_loading import multiframe_tiff
from data_masking import mask_creation
from utils import util


def main():

    path = "/home/maximl/shared_scratch/images"

    # ClientClusterContext creates cluster
    # and registers Client as default client for this session
    with util.ClientClusterContext():
        images = multiframe_tiff.bag_from_directory(path)
        images = images.map(mask_creation.create_mask)
        images.compute()


if __name__ == "__main__":
    main()
