from typing import List, Mapping, Any
from pathlib import Path
from importlib import import_module
import numpy
import dask
import dask.bag


def _substract_mask(event, left_index, right_index, for_channel_index):
    event["mask"][for_channel_index] = event["mask"][left_index] - event["mask"][right_index]
    return event


def segment(
    *,
    images: dask.bag.Bag,
    method: str,
    settings: Mapping[str, Any],
    export: bool,
    output: Path,
    gpu: bool,
    loader_module
) -> dask.bag.Bag:

    mod = import_module('scip.segmentation.%s' % method)

    # this segment operation is annotated with the cellpose resource to let the scheduler
    # know that it should only be executed on a worker that also has the cellpose resource.
    with dask.annotate(resources={"cellpose": 1}):
        images = images.map_partitions(mod.segment_block, gpu_accelerated=gpu > 0, **settings)

    if settings["substract"] is not None:
        images = images.map(
            _substract_mask,
            left_index=settings["substract"]["left_index"],
            right_index=settings["substract"]["right_index"],
            for_channel_index=settings["substract"]["for_index"]
        )

    group_keys = loader_module.get_group_keys()
    if export:
        assert len(group_keys) > 0, "At least one group key is required to export the segmentations"
        assert output is not None, "Output path is required to export the segmentations"
        images = images.map_partitions(
            export_labeled_mask,
            out=output, group_keys=group_keys)

    return images.map_partitions(mod.to_events, group_keys=group_keys, **settings)


def export_labeled_mask(
    events: List[Mapping[str, Any]],
    out: Path,
    group_keys: List[str]
) -> List[Mapping[str, Any]]:
    (out / "masks").mkdir(parents=False, exist_ok=True)

    for event in events:
        f = "%s.npy" % "_".join([str(event[k]) for k in group_keys])
        numpy.save(out / "masks" / f, event["mask"])

    return events
