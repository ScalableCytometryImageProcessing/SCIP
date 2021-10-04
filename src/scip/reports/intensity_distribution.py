import numpy as np
import matplotlib.pyplot as plt
import dask
from io import BytesIO
import base64

from scip.reports.util import get_jinja_template


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
    channels = len(image)

    extent = np.empty(shape=(channels, 2))

    for i in range(channels):
        if len(image[i] > 0):
            extent[i] = [np.min(image[i]), np.max(image[i])]
        else:
            extent[i] = np.nan

    return extent


def reduce_minmax(A, B):
    C = np.concatenate([A[:, 0, np.newaxis], B[:, 0, np.newaxis]], axis=1)
    A[:, 0] = np.nanmin(C, axis=1)
    C = np.concatenate([A[:, 1, np.newaxis], B[:, 1, np.newaxis]], axis=1)
    A[:, 1] = np.nanmax(C, axis=1)
    return A


@dask.delayed
def get_bin_edges(min_max, bin_amount=50):
    """
    Find bin edges using the minimum and maximum of every channel

    Args:
        min_max: array of overall minima and maxima for every channel
        bin_amount (int, optional): amount of bins you want to create. Defaults to 50.

    Returns:
        ndarray: matrix where every row represents the bin edges of a channel
    """

    edges = np.empty(shape=(len(min_max), bin_amount + 1), dtype=float)

    for i in range(len(min_max)):
        if min_max[i, 0] < min_max[i, 1]:
            edges[i] = np.linspace(min_max[i, 0], min_max[i, 1], num=bin_amount + 1)
        else:
            edges[i] = np.linspace(0, 1, num=bin_amount + 1)
    return edges


def get_counts(sample, bins):
    """
    Bin the intensities using the calculated bin edges for every channel

    Args:
        sample (dict): dictionary containing image data
        bins (ndarray): bin edges for every channel

    Returns:
        ndarray: binned intensity counts
    """

    # Count the intensities before masking
    img = sample.get('pixels')
    img = np.reshape(img, newshape=(img.shape[0], -1))
    counts = np.empty(shape=(len(bins), bins.shape[1] - 1), dtype=float)

    # For every channel
    for i in range(len(img)):
        counts[i] = np.histogram(img[i], bins[i])[0]

    return counts


def report(
        bag,
        *,
        template,
        template_dir,
        bin_amount,
        channel_labels,
        output,
        name,
        extent=None
):
    """
    Calculate minima and maxima to find bins, followed by a binning of all
    the intensities. Results are plotted in a report

    Args:
        bag (dask.bag): bag containing dictionaries with image data
        bin_amount (int): number of bins to use for intensity binning
        channel_labels ([str]): names of image channels
        output (str): output file name

    Returns:
        delayed.item: delayed boolean that will be used further in the pipeline
                      to force function execution
    """

    def min_max_partition(part, origin):
        return [get_min_max(p, origin) for p in part]

    def counts_partition(part, bins):
        return [get_counts(p, bins) for p in part]

    @dask.delayed
    def plot_pixel_distribution(counts, bins):

        """
        Plot the intensity distribution for every channel before and after the normalization

        Args:
            counts (ndarray): overall binned intensity counts of pixel data
            bins (ndarray): bin edges for non-normalized data for every channel
            missing_masks (ndarray): count of amount of missing masks for every channel
            output (str): string of output file
        """

        fig, axes = plt.subplots(1, len(channel_labels), figsize=(10, 5))
        for i in range(len(channel_labels)):
            axes[i].title.set_text(channel_labels[i])
            axes[i].bar(
                bins[i, :-1], counts[i], width=(bins[i, -1] - bins[i, 0]) / bin_amount)

        # Encode to include in HTML
        stream = BytesIO()
        fig.savefig(stream, format='png')
        encoded = base64.b64encode(stream.getvalue()).decode('utf-8')

        # Write HTML
        with open(str(output / f"intensity_{name}_quality_control.html"), "w") as fh:
            fh.write(get_jinja_template(template_dir, template).render(name=name, image=encoded))

    if extent is None:
        extent = bag.map_partitions(min_max_partition, origin='flat').fold(reduce_minmax)

    # Get bins from the extrema
    bins = get_bin_edges(extent, bin_amount=bin_amount)

    # Compute the counts
    counts = bag.map_partitions(counts_partition, bins=bins).fold(lambda A, B: A + B)

    @dask.delayed
    def density(n, bins):
        s = n.sum(axis=0)
        s[s == 0] = 1

        # density computation taken from numpy histogram source
        # https://github.com/numpy/numpy/blob/v1.21.0/numpy/lib/histograms.py#L678-L929
        db = np.diff(bins, axis=-1)
        return n / db / s

    counts = dask.delayed(density)(counts, bins)

    # return intensity_count, masked_intensity_count, bins, masked_bins
    plot_pixel_distribution(counts, bins).compute()
