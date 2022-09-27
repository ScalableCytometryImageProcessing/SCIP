from typing import List, Mapping, Any, Optional
from pathlib import Path
from importlib import import_module
import numpy
import dask
import dask.bag
from skimage.measure import regionprops
from scip.utils.util import copy_without


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
    if gpu > 0:
        with dask.annotate(resources={"cellpose": 1}):
            images = images.map_partitions(mod.segment_block, gpu_accelerated=True, **settings)
    else:
        images = images.map_partitions(mod.segment_block, gpu_accelerated=False, **settings)

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

    return images.map_partitions(to_events, group_keys=group_keys, **settings)


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


def to_events(
    events: List[Mapping[str, Any]],
    *,
    group_keys: Optional[List[str]] = None,
    parent_channel_index: int,
    **kwargs
):
    """Converts the segmented objects into a list of dictionaries that can be converted
    to a dask.bag.Bag. The dictionaries contain the pixel information for one detected cell,
    or event, and the meta data of that event.
    """

    newevents = []
    for event in events:

        if group_keys is not None:
            group = "_".join([str(event[k]) for k in group_keys])
        else:
            group = None

        labeled_mask = event["mask"]
        cell_regions = regionprops(labeled_mask[parent_channel_index])

        for props in cell_regions:

            bbox = props.bbox

            mask = labeled_mask[:, bbox[0]:bbox[2], bbox[1]:bbox[3]] == props.label
            combined_mask = numpy.sum(mask, axis=0) > 0

            regions = []
            for m in mask:
                regions.append(int(m.any()))

            newevent = copy_without(event=event, without=["mask", "pixels"])
            newevent["pixels"] = event["pixels"][:, bbox[0]: bbox[2], bbox[1]:bbox[3]]
            newevent["combined_mask"] = combined_mask
            newevent["mask"] = mask
            newevent["group"] = group
            newevent["bbox"] = tuple(bbox)
            newevent["regions"] = regions
            newevent["background"] = numpy.zeros(
                shape=(event["pixels"].shape[0],), dtype=float)
            newevent["combined_background"] = numpy.zeros(
                shape=(event["pixels"].shape[0],), dtype=float)
            newevent["id"] = props.label

            newevents.append(newevent)

    return newevents
