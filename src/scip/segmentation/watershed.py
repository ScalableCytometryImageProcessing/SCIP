from skimage.segmentation import watershed
from skimage.filters import sobel
from skimage import morphology
from skimage.restoration import denoise_nl_means
import numpy
from scip.segmentation import util


def get_mask(el, noisy_channels):

    out = el.copy()

    image = out["pixels"]
    mask = numpy.empty(shape=image.shape, dtype=bool)
    connected_components = []

    for dim in range(len(image)):

        if dim in noisy_channels:
            image[dim] = denoise_nl_means(image[dim], patch_size=2, patch_distance=1)

        elev_map = sobel(image[dim])
        closed = morphology.closing(elev_map, selem=morphology.disk(2))

        markers = numpy.zeros_like(image[dim])
        markers[closed < numpy.quantile(closed, 0.7)] = 1
        markers[closed > numpy.quantile(closed, 0.95)] = 2

        segmentation = watershed(image[dim], markers, compactness=1)

        if segmentation.max() == 0:
            mask[dim] = False
            connected_components.append(0)
        else:
            mask[dim], cc = util.mask_post_process(segmentation == segmentation.max())
            connected_components.append(cc)

    out["intermediate"] = mask
    out["connected_components"] = connected_components

    return out


def create_masks_on_bag(bag, noisy_channels):

    def watershed_masking(partition):
        return [get_mask(p, noisy_channels) for p in partition]

    bag = bag.map_partitions(watershed_masking)
    bag = bag.map_partitions(util.apply_mask_partition)

    return dict(watershed=bag)
