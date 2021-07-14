from skimage import img_as_float, img_as_uint, filters, exposure, morphology, segmentation, measure
from skimage.restoration import denoise_nl_means
from dask.delayed import Delayed
from PIL import Image
import numpy as np
import dask


def denoising(sample: dict[np.ndarray, str]):
    img = sample.get('pixels')
    denoised_masks = np.empty(img.shape, dtype=float)
    channels = img.shape[0]
    a = img[0]
    b = img[1]
    # TODO add list parameter with noisy channels + denoising parameter
    for i in range(channels):
        denoised_masks[i] = denoise_nl_means(img[i], multichannel=False, patch_size=4,
                                            patch_distance=17, fast_mode=True)

    return denoised_masks


def felzenszwalb_segmentation(sample: np.ndarray):
    segmented_masks = np.empty(sample.shape, dtype=float)
    channels = sample.shape[0]
    # TODO add list parameter with felzenszwalb parameters
    for i in range(channels):
        segmented_masks[i] = segmentation.felzenszwalb(sample[i], sigma=0.90, scale=80)
        
    return segmented_masks


def otsu_thresholding(sample: np.ndarray):
    thresholded_masks = np.empty(sample.shape, dtype=bool)
    channels = sample.shape[0]
    # TODO add list parameter with felzenszwalb parameters
    for i in range(channels):
        # Calculation of Otsu threshold
        threshold = filters.threshold_otsu(sample[i])
        
        # Convertion to Boolean mask with Otsu threshold
        thresholded_masks[i] = sample[i] > threshold
        
    return thresholded_masks

def update_dict(sample: np.ndarray, dict_sample: dict[np.ndarray, str]):
    dict_sample = dict_sample.copy()
    dict_sample.update(mask=sample)
    return dict_sample


def create_mask(img):
    denoised = denoising(img)
    segmented = felzenszwalb_segmentation(denoised)
    thresholded = otsu_thresholding(segmented)
    return update_dict(thresholded, img)
