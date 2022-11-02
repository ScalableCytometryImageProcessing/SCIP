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

from typing import Optional, List, Any, Mapping
import numpy
from cellpose import models
from dask.distributed import get_worker
import torch
from distributed import get_client


def segment_block(
    events: List[Mapping[str, Any]],
    *,
    channel_indices: List[int] = None,
    parent_channel_index: int,
    dapi_channel_index: int,
    gpu_accelerated: Optional[bool] = False,
    cell_diameter: Optional[int] = None,
    flow_threshold: Optional[float] = 0.4,
    **kwargs
) -> List[Mapping[str, Any]]:
    """Performs CellPose[1] segmentation.

    [1] Stringer, C., Wang, T., Michaelos, M., & Pachitariu, M. (2021). Cellpose: a generalist
    algorithm for cellular segmentation. Nature methods, 18(1), 100-106.

    Args:
        channel_indices: Indices of channels to be segmented.
        parent_channel_index: Index of parent channel. Objects detected in other
            channels which overlap with an object detected in this channel will be assigned to it.
        dapi_channel_index: Index of DAPI channel.
        gpu_accelerated: Whether segmentation should be run on GPU.
        cell_diameter: See CellPose documentation.
        flow_threshold: See CellPose documentation.
    Returns:
        Events with mask obtained using CellPose.
    """

    if len(events) == 0:
        return events

    w = get_worker()
    if hasattr(w, "cellpose"):
        model = w.cellpose
    else:
        if gpu_accelerated:
            # find all gpu enabled workers
            gpu_workers = [
                address
                for address, w in get_client().scheduler_info()["workers"].items()
                if "cellpose" in w["resources"]
            ]

            gpu_id = gpu_workers.index(w.address)
            device = torch.device(f'cuda:{gpu_id}')
            model = models.Cellpose(gpu=True, device=device, model_type='cyto2')
        else:
            model = models.Cellpose(gpu=False, model_type='cyto2')
        w.cellpose = model

    parents, _, _, _ = model.eval(
        x=[e["pixels"][[parent_channel_index, dapi_channel_index]] for e in events],
        channels=[1, 2],
        diameter=cell_diameter,
        batch_size=128,
        flow_threshold=flow_threshold
    )

    if channel_indices is None:
        channel_indices = range(len(events[0]["pixels"]))

    children = []
    for channel_index in channel_indices:
        if channel_index == parent_channel_index:
            continue

        o, _, _, _ = model.eval(
            x=[e["pixels"][[channel_index, dapi_channel_index]] for e in events],
            channels=[1, 2],
            diameter=cell_diameter,
            batch_size=128,
            flow_threshold=flow_threshold
        )
        children.append((channel_index, o))

    for e_i, event in enumerate(events):

        labeled_mask = numpy.repeat(parents[e_i][numpy.newaxis], event["pixels"].shape[0], axis=0)

        for channel_index, child in children:

            # assign children to parent objects
            mask = numpy.zeros_like(parents[e_i])
            for i in numpy.unique(parents[e_i])[1:]:
                idx, counts = numpy.unique(
                    child[e_i][parents[e_i] == i], return_counts=True)
                idx, counts = idx[1:], counts[1:]  # skip zero (= background)

                idx = idx[(counts / (parents[e_i] == i).sum()) > 0.1]
                mask[numpy.isin(child[e_i], idx) & (parents[e_i] == i)] = i

            labeled_mask[channel_index] = mask

        event["mask"] = labeled_mask

    return events
