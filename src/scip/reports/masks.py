import numpy as np
import matplotlib.pyplot as plt
import dask
from io import BytesIO
import base64


def report(
        bag,
        *,
        channel_labels,
        output,
        name,
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

    def blank_masks_partitions(part):
        return [get_blanks(p) for p in part]

    def get_blanks(sample):
        flat_intensities = sample.get('flat')
        return np.array([len(i) == 0 for i in flat_intensities], dtype=int)

    @dask.delayed
    def plot(percentage):

        # Missing mask bar plot
        fig, ax = plt.subplots()
        ax.bar(channel_labels, percentage)

        # Encode to include in HTML
        stream = BytesIO()
        fig.savefig(stream, format='png')
        encoded = base64.b64encode(stream.getvalue()).decode('utf-8')
        html_missing_masks = '<img src=\'data:image/png;base64,{}\'>'.format(encoded)

        # Write HTML
        with open(str(output / f"{name}_mask_quality_control.html"), "w") as text_file:
            text_file.write('<header><h1>Amount of missing masks per channel</h1></header>')
            text_file.write(html_missing_masks)

    # Calculate the percentage per channel of blank masks
    blanks_sum = bag.map_partitions(blank_masks_partitions).fold(lambda A, B: A + B)

    total = bag.count()
    percentage = dask.delayed(lambda v, t: v / t)(blanks_sum, total)

    plot(percentage)
