from typing import Optional, List
import dask
import numpy
from cellpose import models
import torch
from skimage.measure import regionprops
from dask.distributed import get_worker


@dask.delayed
def segment_block(
    block: numpy.ndarray,
    *,
    group: str,
    idx: int,
    cell_diameter: int,
    dapi_channel_index: Optional[int],
    main_channel_index: int
) -> List[dict]:

    w = get_worker()
    if hasattr(w, "cellpose"):
        model = w.cellpose
    else:
        # gpu_available = torch.cuda.is_available()
        model = models.Cellpose(gpu=False, model_type='cyto2')
        w.cellpose = model

    if dapi_channel_index is None:
        cp_input = block[0, main_channel_index]
        cp_channels = [0,0]
    else:
        cp_input = block[0, [dapi_channel_index, main_channel_index]]
        cp_channels = [1, 2]

    masks, _, _, _ = model.eval(
        x=cp_input,
        channels=cp_channels,
        diameter=cell_diameter
    )

    events = []
    props = regionprops(masks)
    for i, prop in enumerate(props):
        bbox = prop.bbox
        events.append(dict(
            pixels=block[0, :, bbox[0]: bbox[2], bbox[1]:bbox[3]],
            mask=numpy.repeat(prop.image[numpy.newaxis] > 0, block.shape[1], axis=0),
            combined_mask=prop.image > 0,
            idx=idx+i,
            group=group,
            bbox=tuple(bbox),
            regions=[1] * block.shape[1],
            background=numpy.zeros(shape=(block.shape[1],), dtype=float),
            combined_background=numpy.zeros(shape=(block.shape[1],), dtype=float)
        ))

    return events
