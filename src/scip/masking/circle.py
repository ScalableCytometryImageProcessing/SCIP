import numpy
from scip.masking import mask_predicate


def _create_circular_mask(i):

    c, h, w = i.shape

    center = (int(w / 2), int(h / 2))
    radius = min(w, h) // 3

    Y, X = numpy.ogrid[:h, :w]
    dist_from_center = numpy.sqrt((X - center[0])**2 + (Y - center[1])**2)

    mask = dist_from_center <= radius
    return numpy.repeat(mask[numpy.newaxis], repeats=c, axis=0)


def get_mask(el):

    regions = [1] * len(el["pixels"])
    mask = _create_circular_mask(el["pixels"])

    newel = el.copy()
    newel["mask"] = mask
    newel["regions"] = regions
    return newel


def create_masks_on_bag(bag, main_channel):

    def circle_masking(partition):
        return [
            mask_predicate(get_mask(p), main_channel)
            for p in partition
        ]

    bag = bag.map_partitions(circle_masking)
    return bag
