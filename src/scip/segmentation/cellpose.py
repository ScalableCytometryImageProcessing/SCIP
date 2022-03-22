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

from typing import Optional, List
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
    gpu_accelerated: bool,
    cell_diameter: Optional[int],
    segmentation_channel_indices: List[int],
    dapi_channel_index: Optional[int],
    cellpose_segmentation_index: int,
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
    cp_input = block[0, segmentation_channel_indices]
    cells, _, _, _ = model.eval(
        x=cp_input,
        channels=[cellpose_segmentation_index, 0],
        diameter=cell_diameter,
        batch_size=2
    )

    labeled_mask = numpy.repeat(cells[numpy.newaxis], block.shape[1], axis=0)

    if dapi_channel_index is not None:
        # detect nuclei
        cp_input = block[0, dapi_channel_index]
        cp_input = white_tophat(cp_input, footprint=disk(25))
        nuclei, _, _, _ = model.eval(
            x=cp_input,
            channels=[0, 0],
            diameter=cell_diameter,
            batch_size=2
        )

        # assign over-segmented nuclei to parent cells
        for i in numpy.unique(cells)[1:]:
            idx = numpy.unique(nuclei[cells == i])[1:]
            _, counts = numpy.unique(nuclei[cells == i], return_counts=True)
            idx = idx[(counts[1:] / (cells == i).sum()) > 0.1]
            labeled_mask[dapi_channel_index][numpy.isin(nuclei, idx) & (cells == i)] = i

    return labeled_mask


@dask.delayed
def to_events(
    block: numpy.ndarray,
    labeled_mask: numpy.ndarray,
    *,
    segmentation_channel_indices: list[int],
    dapi_channel_index: Optional[int],
    group: str,
    path: str,
    tile: int,
    scene: str,
    **kwargs
):

    cells = labeled_mask[segmentation_channel_indices[0]]
    cell_regions = regionprops(cells)

    events = []
    for props in cell_regions:

        bbox = props.bbox

        mask = labeled_mask[:, bbox[0]:bbox[2], bbox[1]:bbox[3]] == props.label
        combined_mask = cells[bbox[0]:bbox[2], bbox[1]:bbox[3]] == props.label
        regions = [1] * labeled_mask.shape[0]
        if dapi_channel_index is not None:
            regions[dapi_channel_index] = 1 if numpy.any(mask[dapi_channel_index]) else 0

        events.append(dict(
            pixels=block[0, :, bbox[0]: bbox[2], bbox[1]:bbox[3]],
            combined_mask=combined_mask,
            mask=mask,
            group=group,
            bbox=tuple(bbox),
            regions=regions,
            background=numpy.zeros(shape=(block.shape[1],), dtype=float),
            combined_background=numpy.zeros(shape=(block.shape[1],), dtype=float),
            path=path,
            tile=tile,
            scene=scene,
            id=props.label
        ))

    return events
