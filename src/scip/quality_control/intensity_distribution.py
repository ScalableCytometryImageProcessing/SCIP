import numpy as np
import matplotlib.pyplot as plt
import dask
from io import BytesIO
import base64
from scip.data_masking.mask_apply import get_masked_intensities
from datetime import datetime


def get_min_max(sample, origin):

    image = sample.get(origin)

    # pixels values are ndarray, masked_intensities are list(ndarray)
    channels = image.shape[0] if origin == 'pixels' else len(image)

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
    return (np.nanmax(np.vstack((L1[0], L2[0])), axis=0),
            np.nanmin(np.vstack((L1[1], L2[1])), axis=0))


def reduced_counts(L1, L2):
    return (np.array(L1[0]) + np.array(L2[0])), (np.array(L1[1]) + np.array(L2[1]))


@dask.delayed
def get_bins(min_max, bin_amount=50):

    # Rounding errors can result in making bins om n + 1, so we substract small float from n
    n = bin_amount - 0.01
    bins = np.empty(shape=(len(min_max[0]), bin_amount), dtype=float)

    for i in range(len(min_max[0])):

        if min_max[1][i] < min_max[0][i]:
            bins[i] = np.arange(min_max[1][i], min_max[0][i], (min_max[0][i] - min_max[1][i]) / n)
        else:
            bins[i] = np.arange(0, 1, 1 / bin_amount)
    return bins


@dask.delayed
def get_median(stacked_quantiles):

    # Rounding errors can result in making bins om n + 1, so we substract small float from n
    lower = np.nanmedian(stacked_quantiles[0], axis=0)
    upper = np.nanmedian(stacked_quantiles[1], axis=0)

    return (lower, upper)


def get_counts(sample, bins, masked_bins):

    # Count the intensities before masking
    image = sample.get('pixels')
    intensity_count = np.empty((bins.shape[0], bins.shape[1] - 1), dtype=float)
    channels = image.shape[0]

    # For every channel
    for i in range(channels):
        flat = image[i].flatten()
        intensity_count[i] = np.histogram(flat, bins[i])[0]

    # Count the masked intensities
    masked_images = sample.get('masked_intensities')
    masked_intensity_count = np.empty((masked_bins.shape[0], masked_bins.shape[1] - 1), dtype=float)
    channels = len(masked_images)

    # For every channel
    for i in range(channels):
        flat = masked_images[i].flatten()
        masked_intensity_count[i] = np.histogram(flat, masked_bins[i])[0]

    return (intensity_count, masked_intensity_count)


def get_sample_quantile(sample, lower_quantile, upper_quantile, origin):
    image = sample.get(origin)

    channels = image.shape[0] if origin == 'pixels' else len(image)

    lower = np.empty(channels)
    upper = np.empty(channels)

    for i in range(channels):
        if len(image[i] > 0):
            lower[i] = np.quantile(image[i], lower_quantile)
            upper[i] = np.quantile(image[i], upper_quantile)
        else:
            lower[i] = np.NaN
            upper[i] = np.NaN

    return (lower, upper)


def stack_quantiles(L1, L2):
    return (np.vstack((L1[0], L2[0])), np.vstack((L1[1], L2[1])))


def stack_quantiles2(L1, L2):
    return (np.vstack((L1[0], L2[0])), np.vstack((L1[1], L2[1])))


def return_input(L1, L2):
    return (L1, L2)


def get_blanks(sample):
    flat_intensities = sample.get('masked_intensities')
    return np.array([len(i) == 0 for i in flat_intensities], dtype=int)


def reduced_empty_mask(L1, L2):
    return L1 + L2


def segmentation_intensity_report(bag, bin_amount, channels, output):

    def min_max_partition(part, origin):
        return [get_min_max(p, origin) for p in part]

    def masked_intensities_partition(part):
        return [get_masked_intensities(p) for p in part]

    def counts_partition(part, bins, masked_bins):
        return [get_counts(p, bins, masked_bins) for p in part]

    def blank_masks_partitions(part):
        return [get_blanks(p) for p in part]

    @dask.delayed
    def get_percentage(counts, total):
        return counts / total

    # Compute new key-value in dictionary with list of numpy's
    # with flattened intensities per channel
    bag = bag.map_partitions(masked_intensities_partition)

    # Get amount of blank masks
    blanks_sum = bag.map_partitions(blank_masks_partitions).fold(reduced_empty_mask)

    # Calculate the percentage per channel of blank masks
    total = bag.count()
    percentage = get_percentage(blanks_sum, total)

    # Get minima and maxima
    min_max_masked = bag.map_partitions(min_max_partition, origin='masked_img_norm') \
                        .fold(reduced_minmax)
    min_max = bag.map_partitions(min_max_partition, origin='pixels_norm').fold(reduced_minmax)

    # Get bins from the extrema
    bins = get_bins(min_max, bin_amount=bin_amount)
    masked_bins = get_bins(min_max_masked, bin_amount=bin_amount)

    # Compute the counts
    counts = bag.map_partitions(counts_partition, bins=bins, masked_bins=masked_bins) \
        .fold(reduced_counts)

    # return intensity_count, masked_intensity_count, bins, masked_bins
    report_made = plot_before_after_distribution(
        counts, bins, masked_bins, percentage, channels, output)
    return report_made


@dask.delayed
def get_binned_quantile(counts_tuple, bins, masked_bins, lower_quantile, upper_quantile,):
    """
    First method for quantile calculation:
    Based on the bins and bincount we find the bin in which the n% quantile should lie
    """
    def get_single_quantile_value(counts, quantile, bins):
        cum_sum = np.cumsum(counts)
        indices = np.argwhere(cum_sum < counts.sum() * quantile)
        if len(indices) == 0:
            return bins[0]
        return bins[indices[-1][0]]

    counts = counts_tuple[0]
    masked_counts = counts_tuple[1]

    # Unmasked quantile calculation
    lower_bin = [get_single_quantile_value(count, lower_quantile, bin)
                 for count, bin in zip(counts, bins)]
    upper_bin = [get_single_quantile_value(count, upper_quantile, bin)
                 for count, bin in zip(counts, bins)]

    # Masked quantile calculation
    lower_m_bin = [get_single_quantile_value(count, lower_quantile, bin)
                   for count, bin in zip(masked_counts, masked_bins)]
    upper_m_bin = [get_single_quantile_value(count, upper_quantile, bin)
                   for count, bin in zip(masked_counts, masked_bins)]

    return lower_bin, upper_bin, lower_m_bin, upper_m_bin,


def get_distributed_quantile(bag, lower_quantile, upper_quantile):
    """
    Second method for quantile calculation:
    For each sample the quantiles are calculated followed by a reduction over all
    the samples to find median of the quantiles
    """
    def quantile_partition(part, lower_quantile, upper_quantile, origin):

        return [get_sample_quantile(p, lower_quantile, upper_quantile, origin) for p in part]

    def masked_intensities_partition(part):
        return [get_masked_intensities(p) for p in part]

    bag = bag.map_partitions(masked_intensities_partition)

    stacked_quantiles_masked = bag.map_partitions(quantile_partition, lower_quantile,
                                                  upper_quantile, origin="masked_intensities") \
                                  .fold(stack_quantiles)

    stacked_quantiles = bag.map_partitions(quantile_partition, lower_quantile, upper_quantile,
                                           origin="pixels") \
                           .fold(stack_quantiles)

    quantiles = get_median(stacked_quantiles)
    masked_quantiles = get_median(stacked_quantiles_masked)

    return quantiles, masked_quantiles


def get_distributed_partitioned_quantile(bag, lower_quantile, upper_quantile):
    """
    Third method for quantile calculation:
    In every partition intensities are grouped together per channel, on this grouping
    a quantile calculation is performed. The found quantiles per partition are then reduced
    with a median.
    """
    def quantile_partition(part, lower_quantile, upper_quantile, origin):

        collection = []
        if origin == 'pixels':
            channels = part[0].get(origin).shape[0]
        else:
            channels = len(part[0].get(origin))
        channel_quantiles_lower = np.empty(channels)
        channel_quantiles_upper = np.empty(channels)

        if origin == 'pixels':
            channels = part[0].get(origin).shape[0]

            for p in part:
                collection.append(np.vstack([i.flatten() for i in p.get(origin)]))

            flattened_partition = np.hstack(collection)

            channel_quantiles_lower = np.quantile(flattened_partition, lower_quantile, axis=1)
            channel_quantiles_upper = np.quantile(flattened_partition, upper_quantile, axis=1)

        else:
            for i in range(channels):
                channel_collection = [p.get(origin)[i] for p in part]
                stacked_collection = np.hstack(channel_collection)
                if stacked_collection.shape[0] > 0:
                    channel_quantiles_lower[i] = np.quantile(stacked_collection, lower_quantile)
                    channel_quantiles_upper[i] = np.quantile(stacked_collection, upper_quantile)
                else:
                    channel_quantiles_lower[i] = np.nan
                    channel_quantiles_upper[i] = np.nan

        return (channel_quantiles_lower, channel_quantiles_upper)

    def masked_intensities_partition(part):
        return [get_masked_intensities(p) for p in part]

    bag = bag.map_partitions(masked_intensities_partition)

    stacked_quantiles_masked = bag.map_partitions(quantile_partition, lower_quantile,
                                                  upper_quantile, origin="masked_intensities") \
                                  .fold(return_input, stack_quantiles2)

    stacked_quantiles = bag.map_partitions(quantile_partition, lower_quantile, upper_quantile,
                                           origin="pixels").fold(return_input, stack_quantiles2)

    quantiles = get_median(stacked_quantiles)
    masked_quantiles = get_median(stacked_quantiles_masked)

    return quantiles, masked_quantiles


@dask.delayed
def plot_before_after_distribution(counts, bins_before, bins_after,
                                   missing_masks, channels, output, normalize=True, pdf=True):
    counts_before = counts[0]
    counts_after = counts[1]
    if normalize:
        # Normalize the counts so the sum of area is 1
        counts_before = (counts_before.T / (counts_before.sum(axis=1))).T
        counts_after = (counts_after.T / (counts_after.sum(axis=1))).T

    rows = channels
    cols = 2
    intensity_distribution_fg, axarr = plt.subplots(rows, cols, figsize=(20, 30))
    bin_amount = bins_before[0].shape[0]
    for i in range(rows):
        # Plot intensities without mask
        axarr[i, 0].bar(bins_before[i][0:(bin_amount - 1)], counts_before[i],
                        width=(0.01 * (bins_before[i].max() - bins_before[i].min())))

        axarr[i, 0].title.set_text('Before mask')

        # Plot intensities with mask
        axarr[i, 1].bar(bins_after[i][0:(bin_amount - 1)], counts_after[i],
                        width=0.01 * (bins_after[i].max() - bins_after[i].min()))

        axarr[i, 1].title.set_text('After mask')

    # Encode to include in HTML
    distribution_tmp = BytesIO()
    intensity_distribution_fg.savefig(distribution_tmp, format='png')
    encoded = base64.b64encode(distribution_tmp.getvalue()).decode('utf-8')
    html_before_after = '<img src=\'data:image/png;base64,{}\'>'.format(encoded)

    # Missing mask bar plot
    missing_masks_fg = plt.figure()
    channel_labels = [f'ch{i}' for i in range(channels)]
    missing = missing_masks
    plt.bar(channel_labels, missing)

    # Encode to include in HTML
    missing_mask_tmp = BytesIO()
    missing_masks_fg.savefig(missing_mask_tmp, format='png')
    encoded = base64.b64encode(missing_mask_tmp.getvalue()).decode('utf-8')
    html_missing_mask = '<img src=\'data:image/png;base64,{}\'>'.format(encoded)

    # Write HTML
    text_file = open("Intensity_quality_control.html", "w")
    text_file.write('<header><h1>Intensity distribution before vs after masking</h1></header>')
    text_file.write(html_before_after)
    text_file.write('<header><h1>Amount of missing masks per channel</h1></header>')
    text_file.write(html_missing_mask)
    text_file.close()

    return True


def check_report(bag, report_made):

    def check_report(part, report_made):
        if report_made:
            return part

    return bag.map_partitions(check_report, report_made)
