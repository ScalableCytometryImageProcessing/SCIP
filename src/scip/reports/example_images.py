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

import base64
from io import BytesIO
import matplotlib.pyplot as plt
import numpy
import dask

from scip.reports.util import get_jinja_template


def plot_with_masks(images):
    fig = plt.figure(figsize=(10, 5), constrained_layout=False)
    tmp = images[0]["pixels"]
    outer_grid = fig.add_gridspec(len(images), len(tmp), wspace=0.01, hspace=0.01)

    for i in range(len(images)):
        for j in range(len(tmp)):
            inner_grid = outer_grid[i, j].subgridspec(1, 2, wspace=0, hspace=0)
            axes = [fig.add_subplot(inner_grid[0, 0]), fig.add_subplot(inner_grid[0, 1])]
            axes[0].imshow(images[i]["pixels"][j] * images[i]["mask"][j])
            axes[1].imshow(images[i]["mask"][j], cmap="Greys")

            for ax in axes:
                ax.set_axis_off()

    return fig


def plot_no_masks(images):

    fig, axes = plt.subplots(len(images), len(images[0]["pixels"]), squeeze=False, figsize=(10, 5))
    for (i, j), ax in numpy.ndenumerate(axes):
        ax.imshow(images[i]["pixels"][j])
        ax.set_axis_off()

    return fig


@dask.delayed
def report(bag, *, template_dir, template, name, output):

    images = bag[:20]

    if "mask" in images[0]:
        fig = plot_with_masks(images)
    else:
        fig = plot_no_masks(images)

    stream = BytesIO()
    fig.savefig(stream, format='png')
    encoded = base64.b64encode(stream.getvalue()).decode('utf-8')

    # Write HTML
    filename = str("example_images_%s.html" % name)
    with open(str(output / filename), "w") as fh:
        fh.write(get_jinja_template(template_dir, template).render(image=encoded, name=name))

    return True
