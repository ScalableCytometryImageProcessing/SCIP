import base64
from io import BytesIO
import matplotlib.pyplot as plt


def report(bag, *, name, output):

    images = bag.take(3)

    fig, axes = plt.subplots(5, 2)
    fig.suptitle("Images of %s" % name)

    for image, ax in zip(images, axes):
        ax[0].set_axis_off()
        ax[1].set_axis_off()
        ax[0].imshow(image["pixels"][0])
        ax[1].imshow(image["mask"][0])

    stream = BytesIO()
    fig.savefig(stream, format='png')
    encoded = base64.b64encode(stream.getvalue()).decode('utf-8')
    html = '<img src=\'data:image/png;base64,{}\'>'.format(encoded)

    # Write HTML
    with open(str(output / str("mask_%s_quality_control.html" % name)), "w") as text_file:
        text_file.write(
            '<header><h1>Image of %s masks</h1></header>' % name)
        text_file.write(html)
