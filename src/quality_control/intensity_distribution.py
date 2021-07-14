import numpy as np
from PIL import Image
import data_masking.mask_creation 
from skimage import img_as_float, img_as_uint, filters, exposure, morphology, segmentation, measure
from skimage.restoration import denoise_nl_means
import matplotlib.pyplot as plt
from data_masking.mask_apply import get_masked_intensities
from datetime import datetime


def get_min_max(sample, origin):

    image = sample.get(origin)

    # pixels values are ndarray, masked_intensities are list(ndarray)
    channels = image.shape[0] if origin=='pixels' else len(image)

    minima = np.empty(channels)
    maxima = np.empty(channels)

    for i in range(channels):
        if len(image[i] > 0):
            minima[i] = np.amin(image[i])
            maxima[i] = np.amax(image[i])
        else:
            minima[i] = np.NaN
            maxima[i] = np.NaN

    return (maxima, minima)


def reduced_minmax(L1, L2):
    return (np.nanmax(np.vstack((L1[0], L2[0])), axis=0), np.nanmin(np.vstack((L1[1], L2[1])), axis=0))


def reduced_counts(L1, L2):
    return ((np.array(L1[0]) + np.array(L2[0])), (np.array(L1[1]) + np.array(L2[1])), L1[2], L1[3] )


def get_bins(min_max, bin_amount = 50):
    
    # Rounding errors can result in making bins om n + 1, so we substract small float from n
    n = bin_amount - 0.01
    bins = np.empty(shape=(len(min_max[0]),50), dtype=float)

    for i in range(len(min_max[0])):

        if min_max[1][i] < min_max[0][i]:
            bins[i] = np.arange(min_max[1][i], min_max[0][i], (min_max[0][i]-min_max[1][i])/n)
        else:
            bins[i]=np.arange(0, 1, 0.02)
    return bins


def get_counts(sample, bins, masked_bins):

        # Count the intensities before masking
        image = sample.get('pixels')
        intensity_count = np.empty((bins.shape[0], bins.shape[1]-1), dtype=float)
        channels = image.shape[0]

        # For every channel
        for i in range(channels):
            flat = image[i].flatten()
            intensity_count[i] = np.histogram(flat, bins[i])[0]
        
        # Count the masked intensities
        masked_images = sample.get('masked_intensities')
        masked_intensity_count = np.empty((masked_bins.shape[0], masked_bins.shape[1]-1), dtype=float)
        channels = len(masked_images)

        # For every channel
        for i in range(channels):
            flat = masked_images[i].flatten()
            masked_intensity_count[i] = np.histogram(flat, masked_bins[i])[0]

        return (intensity_count, masked_intensity_count, bins, masked_bins)


def get_distributed_counts(bag):
    
    def min_max_partition(part, origin):
        return [get_min_max(p, origin) for p in part]
    
    def masked_intensities_partition(part):
        return [get_masked_intensities(p) for p in part]
    
    def counts_partition(part, bins, masked_bins):
        return [get_counts(p, bins, masked_bins) for p in part]

    # Compute new key-value in dictionary with list of numpy's with flattened intensities per channel
    bag = bag.map_partitions(masked_intensities_partition)

    # Get minima and maxima
    min_max_masked = bag.map_partitions(min_max_partition, 'masked_intensities').fold(reduced_minmax)
    min_max = bag.map_partitions(min_max_partition, 'pixels').fold(reduced_minmax)

    # Get bins from the extrema
    bins = min_max.apply(get_bins)
    masked_bins = min_max_masked.apply(get_bins)

    # Compute the counts 
    intensity_count, masked_intensity_count,  bins, masked_bins = \
        bag.map_partitions(counts_partition, bins=bins, masked_bins=masked_bins).fold(reduced_counts).compute()

    return intensity_count, masked_intensity_count, bins, masked_bins


def plot_before_after_distribution(counts_before, bins_before, counts_after, bins_after, normalize=True, pdf=True):
    
    if normalize:
        # Normalize the counts so the sum of area is 1
        counts_before = (counts_before.T/(counts_before.sum(axis=1))).T
        counts_after = (counts_after.T/(counts_after.sum(axis=1))).T

    rows = 8
    cols = 2
    f, axarr = plt.subplots(rows,cols, figsize=(20, 30))

    for i in range(rows):
        # Plot intensities without mask
        axarr[i,0].bar(bins_before[i][0:49], counts_before[i], width=(0.01*(bins_before[i].max()-bins_before[i].min())))
        axarr[i,0].title.set_text('Before mask')

        # Plot intensities with mask
        axarr[i,1].bar(bins_after[i][0:49], counts_after[i], width=0.01*(bins_after[i].max()-bins_after[i].min()))
        axarr[i,1].title.set_text('After mask')

    # Create a pdf with unique
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y_%H%M%S")
    f.savefig(dt_string+"_intensity_distribution.pdf", bbox_inches='tight')

    return f