import numpy as np
from skimage.measure import regionprops
import numpy
from skimage.morphology import remove_small_objects, label, remove_small_holes
from skimage.segmentation import expand_labels


def mask_predicate(s, bbox_channel):

    # a mask should be present in the bbox_channel
    if not numpy.any(s["mask"][bbox_channel]):
        return False

    # only one connected component should be found in the bbox channel
    if s["regions"][bbox_channel] != 1:
        return False

    return True


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
    background = np.empty(shape=(len(img),), dtype=float)

    # Multiply image with mask to set background to zero
    masked_img = img * mask
    for i in range(img.shape[0]):
        if numpy.any(~mask[i]):
            background[i] = img[i][~mask[i]].mean()
        else:
            background[i] = 0
    
    minr, minc, maxr, maxc = sample["bbox"]

    output = sample.copy()
    output["pixels"] = masked_img[:, minr:maxr, minc:maxc]
    output["mask"] = mask[:, minr:maxr, minc:maxc]
    output["mean_background"] = background.tolist()

    return output


def bounding_box_partition(part, bbox_channel):
    return [get_bounding_box(event, bbox_channel) for event in part]


def get_bounding_box(event, bbox_channel):
    mask = np.where(event["mask"][bbox_channel], 1, 0)
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
