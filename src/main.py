from data_loading import multiframe_tiff
from data_masking import mask_creation
from utils import util
import time


def main():

    start_full = time.time()

    path = "/home/maximl/shared_scratch/images"

    # ClientClusterContext creates cluster
    # and registers Client as default client for this session
    with util.ClientClusterContext(n_workers=24) as context:
        images = multiframe_tiff.bag_from_directory(path)
        images = mask_creation.create_masks_on_bag(images)

        start = time.time()
        images = images.compute()
        print(f"Compute runtime {(time.time() - start):.2f}")
        context.client.profile(filename="profile.html")

    print(f"Full runtime {(time.time() - start_full):.2f}")


if __name__ == "__main__":
    main()
