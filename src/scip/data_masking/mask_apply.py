import numpy as np


def apply(sample, origin):
    """
    Apply binary mask on every channel

    Args:
        dict_sample (dict): dictionary containg image data
        origin (str): key of mask to apply

    Returns:
        dict: dictionary including applied mask
    """

    img = sample.get("pixels")
    mask = sample.get(origin)
    masked_img = np.empty(img.shape, dtype=float)

    # Multiply image with mask to set background to zero
    for i in range(img.shape[0]):
        masked_img[i] = img[i] * mask[i]

    output = sample.copy()
    output["pixels"] = masked_img
    output["mask"] = mask
    return output


def apply_masks_on_bag(bags):
    """
    Apply mask on the image data

    Args:
        bags ({dask.bag}): dict of bags containing dictionaries with pixel and mask data

    Returns:
        [dask.bag]: input bags including applied mask
    """

    def apply_mask_partition(part, origin):
        return [apply_mask(p, origin) for p in part]

    return {k:bag.map_partitions(apply_mask_partition, "result") for k,bag in bags.items()}


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
