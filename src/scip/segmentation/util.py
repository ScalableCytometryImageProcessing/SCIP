from typing import List, Mapping, Any, Optional

from importlib import import_module
from pathlib import Path

import numpy
import dask
import dask.bag
import dask.array


@dask.delayed
def _export_labeled_mask(
    events: List[Mapping[str, Any]],
    output: Path,
    group_keys: List[str]
) -> List[Mapping[str, Any]]:
    (output / "masks").mkdir(parents=False, exist_ok=True)

    for event in events:
        f = "%s.npy" % "_".join([str(event[k]) for k in group_keys])
        numpy.save(output / "masks" / f, event["mask"])

    return events


def bag_from_blocks(
    *,
    blocks: dask.bag.Bag,
    gpu_accelerated: bool,
    segment_method: str,
    segment_kw: Mapping[str, Any],
    output: Optional[Path] = None,
    group_keys: Optional[List[str]] = []
) -> dask.bag.Bag:

    segment_block = import_module('scip.segmentation.%s' % segment_method).segment_block
    to_events = import_module('scip.segmentation.%s' % segment_method).to_events

    # this segment operation is annotated with the cellpose resource to let the scheduler
    # know that it should only be executed on a worker that also has the cellpose resource.
    with dask.annotate(resources={"cellpose": 1}):
        block_events = []
        for block in blocks.to_delayed():
            block_events.append(segment_block(block, gpu_accelerated=gpu_accelerated, **segment_kw))

    if segment_kw["export"]:
        assert len(group_keys) > 0, "At least one group key is required to export the segmentations"
        assert output is not None, "Output path is required to export the segmentations"

        for i in range(len(block_events)):
            block_events[i] = _export_labeled_mask(
                block_events[i], output=output, group_keys=group_keys)

    events = []
    for block in block_events:
        b = to_events(
            block,
            group_keys=group_keys,
            **segment_kw
        )
        events.append(b)

    return dask.bag.from_delayed(events)
