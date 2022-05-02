from typing import List, Mapping, Tuple, Any

from importlib import import_module
from pathlib import Path

import numpy
import dask
import dask.bag
import dask.array
import concurrent.futures


@dask.delayed
def _export_labeled_mask(
    mask: numpy.ndarray,
    output: Path,
    meta: List[Any]
) -> None:
    (output / "masks").mkdir(parents=False, exist_ok=True)
    numpy.save(output / "masks/%s.npy" % "_".join([str(m) for m in meta]), mask)


def bag_from_blocks(
    *,
    blocks: List[dask.array.Array],
    meta: List[Any],
    meta_keys: List[List[Any]],
    gpu_accelerated: bool,
    segment_method: str,
    segment_kw: Mapping[str, Any],
    output: Path
) -> Tuple[dask.bag.Bag, List[concurrent.futures.Future]]:
    
    # blocks = blocks.to_delayed().flatten()

    segment_block = import_module('scip.segmentation.%s' % segment_method).segment_block
    to_events = import_module('scip.segmentation.%s' % segment_method).to_events
    events = []
    futures = []

    if len(meta) == 0:
        meta = [[0]]*len(blocks)

    for m, block in zip(meta, blocks):

        # this segment operation is annotated with the cellpose resource to let the scheduler
        # know that it should only be executed on a worker that also has the cellpose resource.
        with dask.annotate(resources={"cellpose": 1}):
            a = segment_block(
                block,
                gpu_accelerated=gpu_accelerated,
                **segment_kw
            )

        if segment_kw["export"]:
            a = a.persist()
            futures.append(_export_labeled_mask(a, output, m))

        b = to_events(
            block,
            a,
            group="_".join([str(i) for i in m]),
            meta=m,
            meta_keys=meta_keys,
            **segment_kw
        )
        events.append(b)

    bag = dask.bag.from_delayed(events)

    return bag, futures