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
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.morphology import white_tophat, disk


@dask.delayed
def segment_block(
    block: numpy.ndarray,
    *,
    group: str,
    gpu_accelerated: bool,
    cell_diameter: Optional[int],
    dapi_channel_index: Optional[int],
    main_channel_index: int,
    path: str,
    tile: int,
    scene: str,
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

    regions = []

    # detect cells
    cp_input = block[0, main_channel_index]
    sigma = estimate_sigma(cp_input)
    cp_input = denoise_nl_means(
        cp_input, sigma=sigma, h=0.9 * sigma, patch_size=5, patch_distance=5)
    cells, _, _, _ = model.eval(
        x=cp_input,
        channels=[0, 0],
        diameter=cell_diameter,
        batch_size=2
    )
    regions.append(regionprops(cells))

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
        nuclei_mask = numpy.zeros_like(nuclei)
        for i in numpy.unique(cells)[1:]:
            idx = numpy.unique(nuclei[cells == i])[1:]
            _, counts = numpy.unique(nuclei[cells == i], return_counts=True)
            idx = idx[(counts[1:] / (cells == i).sum()) > 0.1]
            nuclei_mask[numpy.isin(nuclei, idx) & (cells == i)] = i
        regions.append(regionprops(nuclei_mask))

    events = []
    for props in zip(*regions):

        bbox = props[0].bbox
        mask = numpy.repeat(props[0].image[numpy.newaxis] > 0, block.shape[1], axis=0)

        if len(props) > 1:
            mask[dapi_channel_index] = nuclei_mask[
                bbox[0]: bbox[2], bbox[1]:bbox[3]] == props[0].label

        events.append(dict(
            pixels=block[0, :, bbox[0]: bbox[2], bbox[1]:bbox[3]],
            combined_mask=props[0].image > 0,
            mask=mask,
            group=group,
            bbox=tuple(bbox),
            regions=[1] * block.shape[1],
            background=numpy.zeros(shape=(block.shape[1],), dtype=float),
            combined_background=numpy.zeros(shape=(block.shape[1],), dtype=float),
            path=path,
            tile=tile,
            scene=scene
        ))

    return events
