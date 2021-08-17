import numpy as np


def apply_mask(dict_sample, origin):
    """
    Apply binary mask on every channel

    Args:
        dict_sample (dict): dictionary containg image data
        origin (str): key of mask to apply

    Returns:
        dict: dictionary including applied mask
    """

    img = dict_sample.get("pixels")
    mask = dict_sample.get(origin)
    masked_img = np.empty(img.shape, dtype=float)

    # Multiply image with mask to set background to zero
    for i in range(img.shape[0]):
        masked_img[i] = img[i] * mask[i]

    return {**dict_sample, **{origin + '_img': masked_img}}


def create_masked_images_on_bag(images):
    """
    Apply both the mask and the largest mask area on the pixel data

    Args:
        images (dask.bag): bag containing dictionaries with pixel and mask data

    Returns:
        dask.bag: input bag including two new key-values: applied mask and applied largest mask area
    """

    def apply_mask_partition(part, origin):
        return [apply_mask(p, origin) for p in part]

    return (
        images
        .map_partitions(apply_mask_partition, 'mask')
        .map_partitions(apply_mask_partition, 'single_blob_mask')
    )


def get_masked_intensities(dict_sample):
    """
    Find the intensities for every channel inside the masks

    Args:
        dict_sample (dict): dictionary containing image data

    Returns:
        dict: dictiory including only the intensities inside mask
    """

    img = dict_sample.get("pixels")
    mask = dict_sample.get("mask")

    masked_intensities = list()

    # Filter the intensities with the mask
    for i in range(img.shape[0]):
        masked_intensities.append(img[i][np.where(mask[i])])

    return {**dict_sample, **dict(masked_intensities=masked_intensities)}
