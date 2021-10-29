import numpy as np
import matplotlib.pyplot as plt
import dask
from io import BytesIO
import base64
from collections import Counter
from scip.reports.util import get_jinja_template


def blank_masks_partitions(part):
    return [get_blanks(p) for p in part]


def get_blanks(s):
    flat = s["mask"].reshape(s["mask"].shape[0], -1)
    return ~np.any(flat, axis=1)


@dask.delayed
def plot1(percentage, channel_labels):
    # Missing mask bar plot
    fig, ax = plt.subplots()
    ax.bar(channel_labels, percentage)
    return fig


@dask.delayed
def plot2(cc_counts, channel_labels):
    fig, axes = plt.subplots(1, len(channel_labels), squeeze=False, sharey=True)
    axes = axes.ravel()
    for counts, ax in zip(cc_counts, axes):
        ax.bar(counts.keys(), counts.values())
    return fig


@dask.delayed
def write_plots(fig1, fig2, name, template_dir, template, output):

    stream = BytesIO()
    fig1.savefig(stream, format='png')
    encoded_missing = base64.b64encode(stream.getvalue()).decode('utf-8')

    stream = BytesIO()
    fig2.savefig(stream, format='png')
    encoded_cc = base64.b64encode(stream.getvalue()).decode('utf-8')

    # Write HTML
    filename = str("mask_quality_control_%s.html" % name)
    with open(str(output / filename), "w") as fh:
        fh.write(get_jinja_template(template_dir, template).render(
            name=name,
            image_missing=encoded_missing,
            image_cc=encoded_cc))

    return True


def report(
        bag,
        *,
        template_dir,
        template,
        name,
        output,
        channel_labels
):
    """
    Calculate minima and maxima to find bins, followed by a binning of all
    the intensities. Results are plotted in a report

    Args:
        bag (dask.bag): bag containing dictionaries with image data
        channels (int): number of image channels
        output (str): output file name
        name (str): name

    Returns:
        delayed.item: delayed boolean that will be used further in the pipeline
                      to force function execution
    """

    # Calculate the percentage per channel of blank masks
    blanks_sum = bag.map_partitions(blank_masks_partitions).fold(lambda A, B: A + B)

    def add_to_count_dict(count_dicts, values):
        for count_dict, value in zip(count_dicts, values):
            count_dict[value] += 1
        return count_dicts

    def merge_count_dicts(l1, l2):
        return [a + b for a, b in zip(l1, l2)]

    cc_counts = bag.map_partitions(lambda part: [p["regions"] for p in part])
    cc_counts = cc_counts.fold(
        binop=add_to_count_dict,
        combine=merge_count_dicts,
        initial=[Counter()] * len(channel_labels)
    )

    total = bag.count()
    percentage = dask.delayed(lambda v, t: v / t)(blanks_sum, total)

    fig1 = plot1(percentage, channel_labels)
    fig2 = plot2(cc_counts, channel_labels)

    return write_plots(
        fig1=fig1,
        fig2=fig2,
        name=name,
        template_dir=template_dir,
        template=template,
        output=output
    )
