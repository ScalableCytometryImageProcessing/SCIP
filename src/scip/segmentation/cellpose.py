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

from typing import Optional, List, Any
import dask
import numpy
from cellpose import models
from skimage.measure import regionprops
from dask.distributed import get_worker
import torch
from skimage.morphology import white_tophat, disk


@dask.delayed
def segment_block(
    block: numpy.ndarray,
    *,
    gpu_accelerated: Optional[bool] = False,
    cell_diameter: Optional[int] = None,
    dapi_channel_index: Optional[int] = None,
    segmentation_channel_index: int,
    **kwargs
) -> List[dict]:

    w = get_worker()
    if hasattr(w, "cellpose"):
        model = w.cellpose
    else:
        if gpu_accelerated:
            device = torch.device(w.name - 2)
            model = models.Cellpose(gpu=True, device=device, model_type='cyto2')
        else:
            model = models.Cellpose(gpu=False, model_type='cyto2')
        w.cellpose = model

    # detect cells
    cp_input = block[segmentation_channel_index]
    cells, _, _, _ = model.eval(
        x=cp_input,
        channels=[0, 0],
        diameter=cell_diameter,
        batch_size=16
    )

    labeled_mask = numpy.repeat(cells[numpy.newaxis], block.shape[0], axis=0)

    if dapi_channel_index is not None:
        # detect nuclei
        cp_input = block[dapi_channel_index]
        cp_input = white_tophat(cp_input, footprint=disk(25))
        nuclei, _, _, _ = model.eval(
            x=cp_input,
            channels=[0, 0],
            diameter=cell_diameter,
            batch_size=16
        )

        # assign over-segmented nuclei to parent cells
        nuclei_mask = numpy.zeros_like(cells)
        for i in numpy.unique(cells)[1:]:
            idx = numpy.unique(nuclei[cells == i])[1:]
            _, counts = numpy.unique(nuclei[cells == i], return_counts=True)
            idx = idx[(counts[1:] / (cells == i).sum()) > 0.1]
            nuclei_mask[numpy.isin(nuclei, idx) & (cells == i)] = i
        labeled_mask[dapi_channel_index] = nuclei_mask

    return labeled_mask


@dask.delayed
def to_events(
    block: numpy.ndarray,
    labeled_mask: numpy.ndarray,
    *,
    dapi_channel_index: Optional[int] = None,
    segmentation_channel_index: int,
    group: str,
    meta: List[Any],
    meta_keys: List[str],
    **kwargs
):
    """Converts the segmented objects into a list of dictionaries that can be converted
    to a dask.bag.Bag. The dictionaries contain the pixel information for one detected cell,
    or event, and the meta data of that event.
    """

    cells = labeled_mask[segmentation_channel_index]
    cell_regions = regionprops(cells)

    events = []
    for props in cell_regions:

        bbox = props.bbox

        mask = labeled_mask[:, bbox[0]:bbox[2], bbox[1]:bbox[3]] == props.label
        combined_mask = cells[bbox[0]:bbox[2], bbox[1]:bbox[3]] == props.label
        regions = [1] * labeled_mask.shape[0]

        if dapi_channel_index is not None:
            regions[dapi_channel_index] = 1 if numpy.any(mask[dapi_channel_index]) else 0

        event = dict(
            pixels=block[:, bbox[0]: bbox[2], bbox[1]:bbox[3]],
            combined_mask=combined_mask,
            mask=mask,
            group=group,
            bbox=tuple(bbox),
            regions=regions,
            background=numpy.zeros(shape=(block.shape[0],), dtype=float),
            combined_background=numpy.zeros(shape=(block.shape[0],), dtype=float),
            id=props.label
        )
        for k, v in zip(meta, meta_keys):
            event[k] = v
        events.append(event)

    return events
