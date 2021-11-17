import numpy
from skimage.morphology import closing, disk
from skimage.filters import threshold_otsu, sobel
from scipy.stats import normaltest
from scip.segmentation import util


def get_mask(el, main, main_channel):

    mask = numpy.empty(shape=el["pixels"].shape, dtype=bool)
    regions = []
    if main:
        x = el["pixels"][main_channel]
        if (normaltest(x.ravel()).pvalue > 0.05):
            # accept H0 that image is gaussian noise = no signal measured
            mask, cc = numpy.zeros(shape=el["pixels"].shape, dtype=bool), 0
        else:
            x = sobel(x)
            x = closing(x, selem=disk(4))
            x = threshold_otsu(x) < x
            for dim in range(len(el["pixels"])):
                mask[dim], cc = util.mask_post_process(x)
                regions.append(cc)
    else:
        for dim in range(len(el["pixels"])):
            if dim == main_channel:
                # in this phase the main channel always has 1 component
                regions.append(1)
                continue

            x = el["pixels"][dim]
            if (normaltest(x.ravel()).pvalue > 0.05):
                # accept H0 that image is gaussian noise = no signal measured
                mask[dim], cc = numpy.zeros(shape=x.shape, dtype=bool), 0
            else:
                x = sobel(x)
                x = closing(x, selem=disk(4))
                x = threshold_otsu(x) < x
                mask[dim], cc = util.mask_post_process(x)
            regions.append(cc)

    out = el.copy()
    out["mask"] = mask
    out["regions"] = regions

    return out


def create_masks_on_bag(bag, main, main_channel):

    def threshold_masking(partition):
        return [get_mask(p, main, main_channel) for p in partition]

    bag = bag.map_partitions(threshold_masking)
    return bag
