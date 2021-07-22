import dask
import dask.bag
from dask.delayed import Delayed
import numpy as np


def apply_mask(dict_sample: dict[np.ndarray, str, np.ndarray], origin):

    img = dict_sample.get("pixels")
    mask = dict_sample.get(origin)
    masked_img = np.empty(img.shape, dtype=float)

    # Multiply image with mask to set background to zero
    for i in range(img.shape[0]):
        masked_img[i] = img[i] * mask[i]

    return {**dict_sample, **{origin + '_img': masked_img}}


def create_masked_images_on_bag(images: dask.bag.Bag):

    def apply_mask_partition(part, origin):
        return [apply_mask(p, origin) for p in part]

    return (
        images
        .map_partitions(apply_mask_partition, 'mask')
        .map_partitions(apply_mask_partition, 'single_blob_mask')
    )


def get_masked_intensities(dict_sample: dict[np.ndarray, str, np.ndarray]):

    img = dict_sample.get("pixels")
    mask = dict_sample.get("mask")

    masked_intensities = list()

    # Filter the intensities with the mask
    for i in range(img.shape[0]):
        masked_intensities.append(img[i][np.where(mask[i])])

    return {**dict_sample, **dict(masked_intensities=masked_intensities)}


def create_masks(imageList: list[Delayed]) -> list[Delayed]:
    mask_samples = []

    for img in imageList:
        mask_samples.append(apply_mask(img))

    return mask_samples
