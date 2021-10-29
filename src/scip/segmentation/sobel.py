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

        segmentation = numpy.full(shape=closed.shape, fill_value=False, dtype=bool)
        segmentation[closed > numpy.quantile(closed, 0.9)] = True

        if segmentation.max() == 0:
            mask[dim] = False
        else:
            segmentation = segmentation == segmentation.max()
            mask[dim] = util.mask_post_process(segmentation)

    out = el.copy()
    out["mask"] = mask

    return out


def create_masks_on_bag(bag, **kwargs):

    def watershed_masking(partition):
        return [get_mask(p) for p in partition]

    bag = bag.map_partitions(watershed_masking)

    return dict(watershed=bag)
