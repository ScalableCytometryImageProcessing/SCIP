import numpy as np
import matplotlib.pyplot as plt
import dask
from io import BytesIO
import base64
from scip.data_masking.mask_apply import get_masked_intensities


def get_min_max(sample, origin):
    """
    Find minima and maxima for every channel

    Args:
        sample (dict): dictionary containing image data
        origin (str): key of image data for which min and max will be calculated

    Returns:
        (ndarray, ndarray): maxima and minima of every channel
    """

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
    """
    Find bin edges using the minimum and maximum of every channel

    Args:
        min_max (ndarray, ndarray): tuple of overall minima and maxima for every channel
        bin_amount (int, optional): amount of bins you want to create. Defaults to 50.

    Returns:
        ndarray: matrix where every row represents the bin edges of a channel
    """

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
def get_median(quantile_vectors):

    lower = np.nanmedian(quantile_vectors[0], axis=0)
    upper = np.nanmedian(quantile_vectors[1], axis=0)

    return (lower, upper)


def get_counts(sample, bins, masked_bins):
    """
    Bin the intensities using the calculated bin edges for every channel, both for
    original pixel data and masked pixel data

    Args:
        sample (dict): dictionary containing image data
        bins (ndarray): bin edges for every channel
        masked_bins (ndarray): bin edges for every channel for the masked images

    Returns:
        (ndarray, ndarray): binned intensity counts for both original and masked image
    """

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
    """
    Find lower and upper quantile values for every channel in an sample

    Args:
        sample (dict): dictionary containing image data
        lower_quantile (float): lower quantile percentage for which we want the value
        upper_quantile (float): higher quantile percentage for which we want the value
        origin (str): key of the data in which to search quantiles

    Returns:
        (ndarray, ndarray): lower and upper quantile values for every channel
    """
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


def return_input(L1, L2):
    return (L1, L2)


def get_blanks(sample):
    flat_intensities = sample.get('masked_intensities')
    return np.array([len(i) == 0 for i in flat_intensities], dtype=int)


def reduced_empty_mask(L1, L2):
    return L1 + L2


def masked_intensities_partition(part):
    return [get_masked_intensities(p) for p in part]


def segmentation_intensity_report(bag, bin_amount, channels, output):
    """
    Calculate minima and maxima to find bins, followed by a binning of all
    the intensities. Results are plotted in a report

    Args:
        bag (dask.bag): bag containing dictionaries with image data
        bin_amount (int): number of bins to use for intensity binning
        channels (int): number of image channels
        output (str): output file name

    Returns:
        delayed.item: delayed boolean that will be used further in the pipeline
                      to force function execution
    """

    def min_max_partition(part, origin):
        return [get_min_max(p, origin) for p in part]

    def counts_partition(part, bins, masked_bins):
        return [get_counts(p, bins, masked_bins) for p in part]

    def blank_masks_partitions(part):
        return [get_blanks(p) for p in part]

    @dask.delayed
    def plot_before_after_distribution(
            counts, bins_before, bins_after, missing_masks, channels,
            output, normalize=True, pdf=True):

        """
        Plot the intensity distribution for every channel before and after the normalization

        Args:
            counts (ndarray, ndarray): overall binned intensities counts of pixel
                                        data before and after
            bins_before (ndarray): bin edges for non-normalized data for every channel
            bins_after (ndarray): bin edges for normalized data for every channel
            missing_masks (ndarray): count of amount of missing masks for every channel
            channels (int): amount of channels
            output (str): string of output file
            normalize (bool, optional): [description]. Defaults to True.
            pdf (bool, optional): [description]. Defaults to True.
        """
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
        with open(str(output / "intensity_quality_control.html"), "w") as text_file:
            text_file.write(
                '<header><h1>Intensity distribution before vs after masking</h1></header>')
            text_file.write(html_before_after)
            text_file.write('<header><h1>Amount of missing masks per channel</h1></header>')
            text_file.write(html_missing_mask)

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
    return plot_before_after_distribution(counts, bins, masked_bins, percentage, channels, output)


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


def get_distributed_partitioned_quantile(bag, lower, upper):
    """
    Third method for quantile calculation:
    In every partition intensities are grouped together per channel, on this grouping
    a quantile calculation is performed. The found quantiles per partition are then reduced
    with a median.
    """

    def select_origin(partition, *, origin):
        """
        Maps each element in the partition to the requested and flattened origin values
        """

        mapped = []
        for el in partition:

            # values cannot be a numpy array as not all channels are required to have the same
            # amount of values for one element (due to masking)
            values = []
            for v in el[origin]:
                values.append(v.flatten())
            mapped.append(values)

        return mapped

    def concatenate_lists(a, b):
        """
        Concatenates the numpy vectors in list a and b element-wise
        """
        for i in range(len(b)):
            a[i] = np.concatenate((a[i], b[i]))

        return a

    def reduce_quantiles(a, b):
        """
        Reduces numpy vectors in lists a and b to their quantiles and concatenates them
        """

        if not hasattr(a, "shape"):
            a = np.array([np.quantile(v, (lower, upper)) for v in a])[..., np.newaxis]
        b = np.array([np.quantile(v, (lower, upper)) for v in b])[..., np.newaxis]
        return np.concatenate([a, b], axis=-1)

    bag = bag.map_partitions(masked_intensities_partition)

    quantiles = bag.map_partitions(select_origin, origin="pixels")
    quantiles = quantiles.fold(concatenate_lists, reduce_quantiles)

    masked_quantiles = bag.map_partitions(select_origin, origin="masked_intensities")
    masked_quantiles = masked_quantiles.fold(concatenate_lists, reduce_quantiles)

    return quantiles, masked_quantiles


def check_report(bag, report_made):
    """
    Dask only executes a function when the returned delayed item is computed or needed
    in the next step of the pipeline. To force the creation of the report we use a simple
    boolean check and return the input partitions forming a bag that will be used in the next steps.

    Args:
        bag (dask.bag): bag containing dictionaries with image data
        report_made (delayed.item): delayed item of a bool

    Returns:
        dask.bag: input bag
    """

    def check_report(part, report_made):
        if report_made:
            return part

    return bag.map_partitions(check_report, report_made)
