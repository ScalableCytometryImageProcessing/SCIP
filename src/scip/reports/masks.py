import numpy as np
import matplotlib.pyplot as plt
import dask
from io import BytesIO
import base64

from scip.reports.util import get_jinja_template

    
def blank_masks_partitions(part):
    return [get_blanks(p) for p in part]

def get_blanks(s):
    flat = s["mask"].reshape(s["mask"].shape[0], -1)
    return ~np.any(flat, axis=1)

@dask.delayed
def plot(percentage, channel_labels):

    # Missing mask bar plot
    fig, ax = plt.subplots()
    ax.bar(channel_labels, percentage)

    return fig

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

    total = bag.count()
    percentage = dask.delayed(lambda v, t: v / t)(blanks_sum, total)

    fig = plot(percentage, channel_labels).compute()

    stream = BytesIO()
    fig.savefig(stream, format='png')
    encoded = base64.b64encode(stream.getvalue()).decode('utf-8')

    # Write HTML
    filename = str("mask_quality_control_%s.html" % name)
    with open(str(output / filename), "w") as fh:
        fh.write(get_jinja_template(template_dir, template).render(image=encoded, name=name))
