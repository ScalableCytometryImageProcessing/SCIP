import numpy as np
from skimage.measure import regionprops
import numpy
from skimage.morphology import closing, remove_small_objects, label, disk, remove_small_holes


def mask_predicate(s):

    if not all(c == 1 for c in s["connected_components"]):
        return False

    flat = s["mask"].reshape(s["mask"].shape[0], -1)
    return all(numpy.any(flat, axis=1))


def apply_mask_partition(part):
    return [apply(p, "mask") for p in part]


def apply(sample, origin):
    """
    Apply binary mask on every channel

    Args:
        dict_sample (dict): dictionary containg image data
        origin (str): key of mask to apply

    Returns:
        dict: dictionary including applied mask
    """

    img = sample["pixels"]
    mask = sample[origin]
    masked_img = np.empty(img.shape, dtype=float)

    # Multiply image with mask to set background to zero
    for i in range(img.shape[0]):
        masked_img[i] = img[i] * mask[i]

    output = sample.copy()
    output["pixels"] = masked_img
    output["mask"] = mask
    return output


def crop_to_mask_partition(part):
    return [crop_to_mask(p) for p in part]


def crop_to_mask(sample):

    minr, minc, maxr, maxc = sample["bbox"]

    newsample = sample.copy()
    newsample["pixels"] = sample["pixels"][:, minr:maxr, minc:maxc]
    newsample["mask"] = sample["mask"][:, minr:maxr, minc:maxc]

    return newsample


def bounding_box_partition(part):
    return [get_bounding_box(event) for event in part]


def get_bounding_box(event):
    mask = np.where(event["mask"], 1, 0)

    if numpy.any(mask[0]):
        bbox = list(regionprops(mask[0])[0].bbox)
        for m in mask:
            if not numpy.any(m):
                bbox = None, None, None, None
                break

            tmp = regionprops(m)[0].bbox

            bbox[0] = min(bbox[0], tmp[0])
            bbox[1] = min(bbox[1], tmp[1])
            bbox[2] = max(bbox[2], tmp[2])
            bbox[3] = max(bbox[3], tmp[3])
    else:
        bbox = None, None, None, None

    newevent = event.copy()
    newevent["bbox"] = tuple(bbox)

    return newevent


def mask_post_process(mask):
    mask = remove_small_holes(mask, area_threshold=300)
    mask = remove_small_objects(mask, min_size=30)
    mask = label(mask)
    
    return mask > 0, mask.max()
