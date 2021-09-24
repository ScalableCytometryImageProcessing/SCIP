import numpy as np
from skimage.measure import regionprops
import numpy


def nonempty_mask_predicate(s):
    flat = s["mask"].reshape(s["mask"].shape[0], -1)
    return all(numpy.any(flat, axis=1))


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

    minr, minc, maxr, maxc = d["bbox"]

    d = d.copy()
    d["pixels"] = d["pixels"][:, minr:maxr, minc:maxc]
    d["mask"] = d["mask"][:, minr:maxr, minc:maxc]

    return d


def bounding_box_partition(part):
    return [get_bounding_box(event) for event in part]


def get_bounding_box(event):
    mask = np.where(event["mask"], 1, 0)
    bbox = [event["pixels"].shape[1], event["pixels"].shape[2], 0, 0]
    for m in mask:
        if not numpy.any(m):
            bbox = None, None, None, None
            break

        prop = regionprops(m)[0]
        tmp = prop.bbox

        bbox[0] = min(bbox[0], tmp[0])
        bbox[1] = min(bbox[1], tmp[1])
        bbox[2] = max(bbox[2], tmp[2])
        bbox[3] = max(bbox[3], tmp[3])

    event = event.copy()
    event["bbox"] = tuple(bbox)

    return event
