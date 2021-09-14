from skimage.segmentation import watershed
from skimage.filters import sobel
from skimage import morphology
import numpy
from scip.segmentation import mask_apply


def get_mask(el):

    image = el["pixels"]
    mask = numpy.empty(shape=image.shape, dtype=bool)

    for dim in range(len(image)):
        # compute elevation map
        elev_map = sobel(image[dim])

        # select watershed markers
        markers = numpy.zeros_like(image[1])
        markers[image[1] < numpy.quantile(image[dim], 0.05)] = 1
        markers[image[1] > numpy.quantile(image[dim], 0.95)] = 2

        # do watershed
        segmentation = watershed(elev_map, markers)

        # binarize
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
    bag = bag.map_partitions(mask_apply.apply_mask_partition)

    return dict(watershed=bag)
