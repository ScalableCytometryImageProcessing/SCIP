import numpy as np
from skimage.measure import regionprops
import numpy
from skimage.morphology import remove_small_objects, label, remove_small_holes
from skimage.segmentation import expand_labels

from scip.utils.util import copy_without, check


def mask_predicate(s, bbox_channel_index):

    # a mask should be present in the bbox_channel_index
    # only one connected component should be found in the bbox channel
    if (
        (not numpy.any(s["mask"][bbox_channel_index])) or (s["regions"][bbox_channel_index] != 1)
    ):
        return copy_without(s, without=["mask", "pixels"])

    return s


def apply_mask_partition(part):
    return [apply(p) for p in part]


@check
def apply(sample):
    """
    Apply binary mask on every channel

    Args:
        dict_sample (dict): dictionary containg image data
        origin (str): key of mask to apply

    Returns:
        dict: dictionary including applied mask
    """

    img = sample["pixels"]
    mask = sample["mask"]
    combined_mask = numpy.sum(mask, axis=0) > 0
    background = np.empty(shape=(len(img),), dtype=float)
    combined_background = np.empty(shape=(len(img),), dtype=float)

    # Multiply image with mask to set background to zero
    for i in range(img.shape[0]):
        if numpy.any(~mask[i]):
            background[i] = img[i][~mask[i]].mean()
        else:
            background[i] = 0
        combined_background[i] = img[i][~combined_mask].mean()

    minr, minc, maxr, maxc = sample["bbox"]

    output = copy_without(sample, ["pixels", "mask"])
    output["pixels"] = img[:, minr:maxr, minc:maxc]
    output["mask"] = mask[:, minr:maxr, minc:maxc]
    output["combined_mask"] = combined_mask[minr:maxr, minc:maxc]
    output["background"] = background.tolist()
    output["combined_background"] = combined_background.tolist()

    return output


def bounding_box_partition(part, bbox_channel_index):
    return [get_bounding_box(event, bbox_channel_index) for event in part]


@check
def get_bounding_box(event, bbox_channel_index):
    mask = np.where(event["mask"][bbox_channel_index], 1, 0)
    bbox = list(regionprops(mask)[0].bbox)

    newevent = event.copy()
    newevent["bbox"] = tuple(bbox)

    return newevent


def mask_post_process(mask):
    mask = remove_small_holes(mask, area_threshold=300)
    mask = expand_labels(label(mask), distance=1)
    mask = remove_small_objects(mask > 0, min_size=20)
    mask = label(mask)

    return mask > 0, mask.max()
