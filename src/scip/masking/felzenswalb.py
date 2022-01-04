# Copyright (C) 2022 Maxim Lippeveld
#
# This file is part of SCIP.
#
# SCIP is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SCIP is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with SCIP.  If not, see <http://www.gnu.org/licenses/>.

from skimage import filters, segmentation
from skimage.restoration import denoise_nl_means
from skimage.measure import label, regionprops
import numpy as np


def denoising(sample, noisy_channels=[]):
    """
    Non-local mean denoising

    Args:
        sample (dict): dictionary containing pixel data of an image
        noisy_channels (list, optional): list of channels with lots of noise, non-local mean
         denoising won't be used for these. Defaults to [].

    Returns:
        dict: input dictionary including the denoised pixel data
    """
    img = sample.get('pixels')
    denoised = np.empty(img.shape, dtype=float)
    channels = img.shape[0]
    for i in range(channels):
        if i not in noisy_channels:
            denoised[i] = denoise_nl_means(
                img[i], multichannel=False, patch_size=4,
                patch_distance=17, fast_mode=True)
        else:
            denoised[i] = img[i]

    sample["intermediate"] = denoised
    return sample


def felzenszwalb_segmentation(sample):
    """
    Felzenszwalb segmentation

    Args:
        sample (dict): dictionary containing pixel data, paths and denoised pixel data

    Returns:
        dict: input dictionary including the segmentation pixel data
    """
    segmented = np.empty(sample["intermediate"].shape, dtype=float)
    channels = sample["intermediate"].shape[0]
    # TODO add list parameter with felzenszwalb parameters
    for i in range(channels):
        segmented[i] = segmentation.felzenszwalb(sample["intermediate"][i], sigma=0.90, scale=80)

    sample["intermediate"] = segmented
    return sample


def otsu_thresholding(sample):
    """
    Otsu thresholding

    Args:
        sample (dict): dictionary containing pixel data, paths, denoised and segmented data

    Returns:
        dict: input dictionary including masks for every channel
    """
    thresholded_masks = np.empty(sample["intermediate"].shape, dtype=bool)
    channels = sample["intermediate"].shape[0]
    for i in range(channels):
        # Calculation of Otsu threshold
        threshold = filters.threshold_otsu(sample["intermediate"][i])

        # Convertion to Boolean mask with Otsu threshold
        thresholded_masks[i] = sample["intermediate"][i] > threshold

    sample["intermediate"] = thresholded_masks
    return sample


def largest_blob_detection(sample: dict):
    """
    Detection of the largest fully connected mask region for every channel.

    Args:
        sample (dict): image dictionary containing mask data

    Returns:
        dict: input dictionary including largest mask region for every channel

    """

    def largest_region(regions):
        largest = 0
        largest_index = 0
        for props in regions:
            if props.area > largest:
                largest = props.area
                largest_index = regions.index(props)
        return largest_index

    largest_blob = np.empty(sample["intermediate"].shape, dtype=float)
    channels = sample["intermediate"].shape[0]
    for i in range(channels):
        label_img = label(sample["intermediate"][i])
        regions = regionprops(label_img)
        if len(regions) == 0:
            largest_blob[i] = sample["intermediate"][i]
        else:
            largest_blob[i] = np.where(label_img == (largest_region(regions) + 1), 1, 0)

    sample["mask"] = largest_blob.astype(bool)
    return sample


def create_masks_on_bag(images, noisy_channels):
    """
    Create masks and largest region masks on every image in the bag

    Args:
        images (dask.bag): bag containing dictionaries with pixel data and paths
        noisy_channels (list): list of channels with lots of noise, non-local mean denoising
                               won't be used for these.

    Returns:
        dask.bag: bag containing dictionaries including masks and largest region masks
                 (intermediate masking data is also included for now)
    """

    # we define the different steps as named functions
    # so that Dask can differentiate between them in
    # the dashboard

    def denoise_partition(part):
        return [denoising(p, noisy_channels) for p in part]

    def felzenswalb_segment_partition(part):
        return [felzenszwalb_segmentation(p) for p in part]

    def otsu_threshold_partition(part):
        return [otsu_thresholding(p) for p in part]

    def largest_blob_partition(part):
        return [largest_blob_detection(p) for p in part]

    otsu = (
        images
        .map_partitions(denoise_partition)
        .map_partitions(felzenswalb_segment_partition)
        .map_partitions(otsu_threshold_partition)
        .map_partitions(largest_blob_partition)
    )

    return otsu
