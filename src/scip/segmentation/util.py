import numpy as np
from skimage.measure import regionprops


def apply_mask_partition(part):
    return [apply(p, "intermediate") for p in part]


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
    del output["intermediate"]
    output["pixels"] = masked_img
    output["mask"] = mask
    return output


def masked_intensities_partition(part):
    return [get_masked_intensities(p) for p in part]


def get_masked_intensities(sample):
    """
    Find the intensities for every channel inside the masks

    Args:
        sample (dict): dictionary containing image data

    Returns:
        dict: dictiory including flatmask (only the intensities inside mask)
    """

    img = sample.get("pixels")
    mask = sample.get("mask")

    masked_intensities = list()

    # Filter the intensities with the mask
    for i in range(len(img)):
        masked_intensities.append(img[i][mask[i]])

    output = sample.copy()
    output["flat"] = masked_intensities
    return output


def crop_to_mask_partition(part):
    return [crop_to_mask(p) for p in part]


def crop_to_mask(d):

    d = d.copy()

    mask = np.where(d["mask"], 1, 0)
    minr, minc, maxr, maxc = d["pixels"].shape[1], d["pixels"].shape[2], 0, 0
    for m in mask:
        prop = regionprops(m)[0]
        bbox = prop.bbox

        minr = min(bbox[0], minr)
        minc = min(bbox[1], minc)
        maxr = max(bbox[2], maxr)
        maxc = max(bbox[3], maxc)

    d["pixels"] = d["pixels"][:, minr:maxr, minc:maxc]
    d["mask"] = d["mask"][:, minr:maxr, minc:maxc]

    return d
