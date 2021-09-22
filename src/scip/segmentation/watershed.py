from skimage.segmentation import watershed
from skimage.filters import sobel
from skimage import morphology
import numpy
from scip.segmentation import util


def get_mask(el):

    image = el["pixels"]
    mask = numpy.empty(shape=image.shape, dtype=bool)

    for dim in range(len(image)):
        # compute elevation map
        elev_map = sobel(image[dim])
        closed = morphology.closing(elev_map, selem=morphology.disk(2))

        # select watershed markers
        markers = numpy.zeros_like(image[1])
        markers[image[1] < numpy.quantile(image[dim], 0.05)] = 1
        markers[image[1] > numpy.quantile(image[dim], 0.95)] = 2

        # do watershed
        segmentation = watershed(closed, markers)

        # binarize
        if segmentation.max() == 0:
            mask[dim] = False
        else:
            segmentation = segmentation == segmentation.max()

            # post process segmentation
            segmentation = morphology.binary_dilation(segmentation, selem=morphology.disk(2))

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
