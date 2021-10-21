import numpy
from skimage.morphology import closing, disk
from skimage.filters import threshold_otsu, sobel

from scip.segmentation import util


def get_mask(el):

    mask = numpy.empty(shape=el["pixels"].shape, dtype=bool)
    connected_components = []

    for dim in range(len(el["pixels"])):

        x = el["pixels"][dim] 
        x = sobel(x)
        x = closing(x, selem=disk(4))
        x = threshold_otsu(x) < x

        mask[dim], cc = util.mask_post_process(x)
        connected_components.append(cc)

    out = el.copy()
    out["mask"] = mask
    out["connected_components"] = connected_components

    return out


def create_masks_on_bag(bag, **kwargs):

    def threshold_masking(partition):
        return [get_mask(p) for p in partition]

    bag = bag.map_partitions(threshold_masking)
    return bag
