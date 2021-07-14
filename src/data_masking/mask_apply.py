from dask.delayed import Delayed
import numpy as np


def apply_mask(dict_sample: dict[np.ndarray, str, np.ndarray]):
    dict_sample = dict_sample.copy()
    img = dict_sample.get("image")
    mask = dict_sample.get("mask")
    masked_img = np.empty(img.shape, dtype=float)

    for i in range(img.shape[0]):
        masked_img[i] = img[i] * mask[i]

    dict_sample.update(masked_img=masked_img)
    return dict_sample


def get_masked_intensities(dict_sample: dict[np.ndarray, str, np.ndarray]):

    # Make a copy of the dict, input parameters in Dask shouldn't be changed
    dict_sample = dict_sample.copy()

    img = dict_sample.get("pixels")
    mask = dict_sample.get("mask")

    masked_intensities = list()

    # Flatten and filter the intensities with the mask
    for i in range(img.shape[0]):
        img_flatten = img[i].flatten()
        mask_flatten = mask[i].flatten()
        masked_intensities.append(np.extract(mask_flatten, img_flatten))

    # Update dictionary with new key-value
    dict_sample.update(masked_intensities=masked_intensities)
    return dict_sample


def create_masks(imageList: list[Delayed]) -> list[Delayed]:
    mask_samples = []

    for img in imageList:
        mask_samples.append(apply_mask(img))

    return mask_samples
