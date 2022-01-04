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

import numpy as np
import matplotlib.pyplot as plt
import dask
from io import BytesIO
import base64
from scip.normalization.quantile_normalization import get_distributed_minmax

from scip.reports.util import get_jinja_template


@dask.delayed
def get_bin_edges(min_maxes, bin_amount=20):
    """
    Find bin edges using the minimum and maximum of every channel

    Args:
        min_max: array of overall minima and maxima for every channel for every group
        bin_amount (int, optional): amount of bins you want to create. Defaults to 50.

    Returns:
        ndarray: matrix where every row represents the bin edges of a channel
    """

    out = {}
    for k, min_max in min_maxes:
        edges = np.empty(shape=(len(min_max), bin_amount + 1), dtype=float)

        for i in range(len(min_max)):
            if min_max[i, 0] < min_max[i, 1]:
                edges[i] = np.linspace(min_max[i, 0], min_max[i, 1], num=bin_amount + 1)
            else:
                edges[i] = np.linspace(0, 1, num=bin_amount + 1)
        out[k] = edges
    return out


def get_counts(sample, bins_per_group):
    """
    Bin the intensities using the calculated bin edges for every channel

    Args:
        sample (dict): dictionary containing image data
        bins (ndarray): bin edges for every channel

    Returns:
        ndarray: binned intensity counts
    """

    # Count the intensities before masking
    bins = bins_per_group[sample["group"]]

    if "mask" in sample:
        img = [sample["pixels"][i][sample["mask"][i]] for i in range(len(sample["pixels"]))]
    else:
        img = sample["pixels"]
    counts = np.empty(shape=(len(bins), bins.shape[1] - 1), dtype=float)

    # For every channel
    for i in range(len(img)):
        counts[i] = np.histogram(img[i].flatten(), bins[i])[0]

    out = {}
    out[sample["group"]] = counts
    return out


def report(
        bag,
        *,
        template,
        template_dir,
        bin_amount,
        channel_names,
        output,
        name
):
    """
    Calculate minima and maxima to find bins, followed by a binning of all
    the intensities. Results are plotted in a report

    Args:
        bag (dask.bag): bag containing dictionaries with image data
        bin_amount (int): number of bins to use for intensity binning
        channel_names ([str]): names of image channels
        output (str): output file name

    Returns:
        delayed.item: delayed boolean that will be used further in the pipeline
                      to force function execution
    """

    def counts_partition(part, bins):
        return [get_counts(p, bins) for p in part]

    @dask.delayed
    def plot_pixel_distribution(counts_per_group, bins_per_group):

        """
        Plot the intensity distribution for every channel before and after the normalization

        Args:
            counts (ndarray): overall binned intensity counts of pixel data
            bins (ndarray): bin edges for non-normalized data for every channel
            missing_masks (ndarray): count of amount of missing masks for every channel
            output (str): string of output file
        """

        fig, axes = plt.subplots(
            len(counts_per_group), len(channel_names), squeeze=False, figsize=(10, 5))
        for i, k in enumerate(counts_per_group.keys()):
            counts = counts_per_group[k]
            bins = bins_per_group[k]

            for j in range(len(channel_names)):
                axes[i, j].set_title(channel_names[j])
                axes[i, j].bar(
                    bins[j, :-1], counts[j], width=(bins[j, -1] - bins[j, 0]) / bin_amount)

        # Encode to include in HTML
        stream = BytesIO()
        fig.savefig(stream, format='png')
        encoded = base64.b64encode(stream.getvalue()).decode('utf-8')

        # Write HTML
        with open(str(output / f"intensity_{name}_quality_control.html"), "w") as fh:
            fh.write(get_jinja_template(template_dir, template).render(name=name, image=encoded))

        return True

    extent = get_distributed_minmax(bag, len(channel_names))

    # Get bins from the extrema
    bins = get_bin_edges(extent, bin_amount=bin_amount)

    # Compute the counts
    def merge_dicts(a, b):
        keys = set(a.keys()).union(b.keys())
        result = {}
        for k in keys:
            if (k in a) and (k in b):
                result[k] = a[k] + b[k]
            elif k in a:
                result[k] = a[k]
            else:
                result[k] = b[k]
        return result

    counts = bag.map_partitions(counts_partition, bins=bins).fold(
        binop=merge_dicts,
        combine=merge_dicts,
        initial=dict()
    )

    # @dask.delayed
    # def density(n, bins):
    #     s = n.sum(axis=0)
    #     s[s == 0] = 1

    #     # density computation taken from numpy histogram source
    #     # https://github.com/numpy/numpy/blob/v1.21.0/numpy/lib/histograms.py#L678-L929
    #     db = np.diff(bins, axis=-1)
    #     return n / db / s

    # counts = dask.delayed(density)(counts, bins)

    return plot_pixel_distribution(counts, bins)
