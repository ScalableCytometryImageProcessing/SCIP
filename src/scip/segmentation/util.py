from typing import List, Mapping, Any

from importlib import import_module
from pathlib import Path

import numpy
import dask
import dask.bag
import dask.array


@dask.delayed
def _export_labeled_mask(
    mask: numpy.ndarray,
    output: Path,
    meta: List[Any]
) -> numpy.ndarray:
    (output / "masks").mkdir(parents=False, exist_ok=True)
    numpy.save(output / "masks" / ("%s.npy" % "_".join([str(m) for m in meta])), mask)
    return mask


def bag_from_blocks(
    *,
    blocks: List[dask.array.Array],
    paths: List[str],
    meta_keys: List[str],
    meta: List[List[Any]],
    gpu_accelerated: bool,
    segment_method: str,
    segment_kw: Mapping[str, Any],
    output: Path
) -> dask.bag.Bag:

    segment_block = import_module('scip.segmentation.%s' % segment_method).segment_block
    to_events = import_module('scip.segmentation.%s' % segment_method).to_events
    events = []

    if len(meta) == 0:
        meta = [[i] for i in range(len(blocks))]
        meta_keys = ["block"]

    for path, m, block in zip(paths, meta, blocks):

        # this segment operation is annotated with the cellpose resource to let the scheduler
        # know that it should only be executed on a worker that also has the cellpose resource.
        with dask.annotate(resources={"cellpose": 1}):
            a = segment_block(
                block,
                gpu_accelerated=gpu_accelerated,
                **segment_kw
            )

        if segment_kw["export"]:
            a = _export_labeled_mask(a, output, m)

        b = to_events(
            block,
            a,
            group="_".join([str(i) for i in m]),
            meta=m + [path],
            meta_keys=meta_keys + ["path"],
            **segment_kw
        )
        events.append(b)

    bag = dask.bag.from_delayed(events)

    return bag
