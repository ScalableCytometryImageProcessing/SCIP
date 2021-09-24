from skimage.segmentation import watershed
from skimage.filters import sobel
from skimage import morphology
import numpy
from scip.segmentation import util


def get_mask(el):

    image = el["pixels"]
    mask = numpy.empty(shape=image.shape, dtype=bool)

    for dim in range(len(image)):

        elev_map = sobel(image[dim])
        closed = morphology.closing(elev_map, selem=morphology.disk(2))

        markers = numpy.zeros_like(closed)
        markers[closed < numpy.quantile(closed, 0.1)] = 1
        markers[closed > numpy.quantile(closed, 0.9)] = 2

        segmentation = watershed(closed, markers)

        if segmentation.max() == 0:
            mask[dim] = False
        else:
            segmentation = segmentation == segmentation.max()
            segmentation = morphology.binary_closing(segmentation, selem=morphology.disk(2))
            mask[dim] = segmentation

    out = el.copy()
    out["intermediate"] = mask

    return out


def create_masks_on_bag(bag, **kwargs):

    def watershed_masking(partition):
        return [get_mask(p) for p in partition]

    bag = bag.map_partitions(watershed_masking)
    bag = bag.map_partitions(util.apply_mask_partition)

    return dict(watershed=bag)
