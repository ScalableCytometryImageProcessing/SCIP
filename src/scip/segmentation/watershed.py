from skimage.segmentation import watershed
from skimage.filters import sobel, threshold_otsu
from skimage import morphology
from skimage.restoration import denoise_nl_means
import numpy
from scip.segmentation import util


def get_mask(el, noisy_channels):

    out = el.copy()

    image = out["pixels"]
    mask = numpy.empty(shape=image.shape, dtype=bool)
    regions = []

    for dim in range(len(image)):

        if dim in noisy_channels:
            image[dim] = denoise_nl_means(image[dim], patch_size=2, patch_distance=1)

        elev_map = sobel(image[dim])
        closed = morphology.closing(elev_map, selem=morphology.disk(2))

        # markers = numpy.zeros_like(image[dim])
        # markers[closed < numpy.quantile(closed, 0.7)] = 1
        # markers[closed > numpy.quantile(closed, 0.95)] = 2
        markers = numpy.zeros_like(image[dim])
        thresh = threshold_otsu(closed)
        markers[closed < thresh - thresh * 0.5] = 1
        markers[closed > thresh + thresh * 0.5] = 2

        segmentation = watershed(image[dim], markers, compactness=1)

        if segmentation.max() == 0:
            mask[dim] = False
            regions.append(0)
        else:
            mask[dim], cc = util.mask_post_process(segmentation == segmentation.max())
            regions.append(cc)

    out["mask"] = mask
    out["regions"] = regions

    return out


def create_masks_on_bag(bag, noisy_channels):

    def watershed_masking(partition):
        return [get_mask(p, noisy_channels) for p in partition]

    bag = bag.map_partitions(watershed_masking)

    return bag
