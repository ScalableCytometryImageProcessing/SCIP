from skimage import filters, segmentation
from skimage.restoration import denoise_nl_means
import numpy as np
import dask
import dask.bag


def denoising(sample: dict[np.ndarray, str]):
    img = sample.get('pixels')
    denoised_masks = np.empty(img.shape, dtype=float)
    channels = img.shape[0]
    # TODO add list parameter with noisy channels + denoising parameter
    for i in range(channels):
        denoised_masks[i] = denoise_nl_means(
            img[i], multichannel=False, patch_size=4,
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


def create_masks_on_bag(images: dask.bag.Bag):

    # we define the different steps as named functions
    # so that Dask can differentiate between them in
    # the dashboard

    def denoise_partition(part):
        return [denoising(p) for p in part]

    def felzenswalb_segment_partition(part):
        return [felzenszwalb_segmentation(p) for p in part]

    def otsu_threshold_partition(part):
        return [otsu_thresholding(p) for p in part]

    return (
        images
        .map_partitions(denoise_partition)
        .map_partitions(felzenswalb_segment_partition)
        .map_partitions(otsu_threshold_partition)
    )
