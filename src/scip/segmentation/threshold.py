import numpy
from skimage.morphology import closing, disk, remove_small_holes, remove_small_objects, label
from skimage.filters import threshold_otsu, sobel, gaussian
from scipy.stats import normaltest
from scip.segmentation import util


def get_mask(el, main, main_channel):

    if main:
        regions = [0] * len(el["pixels"])
        mask, cc = numpy.full(shape=el["pixels"].shape, dtype=bool, fill_value=False), 0
        x = el["pixels"][main_channel]
        if (normaltest(x.ravel()).pvalue < 0.05):
            x = sobel(x)
            x = closing(x, selem=disk(4))
            x = threshold_otsu(x) < x
            mask[main_channel], cc = util.mask_post_process(x)
        regions[main_channel] = cc
    else:
        regions = []
        # search for objects within the bounding box found on the main_channel
        mask = el["mask"]
        bbox = el["bbox"]
        for dim in range(len(el["pixels"])):
            if dim == main_channel:
                # in this phase the main channel always has 1 component
                regions.append(1)
                continue

            x = el["pixels"][dim, bbox[0]:bbox[2], bbox[1]:bbox[3]]
            x = gaussian(x, sigma=1)
            if (normaltest(x.ravel()).pvalue > 0.05):
                # accept H0 that image is gaussian noise = no signal measured
                mask[dim], cc = numpy.zeros(shape=el["pixels"][dim].shape, dtype=bool), 0
            else:
                x = sobel(x)
                x = closing(x, selem=disk(2))
                x = threshold_otsu(x) < x
                x[[0, -1], :] = 0
                x[:, [0, -1]] = 0
                x = remove_small_holes(x, area_threshold=40)
                x = remove_small_objects(x, min_size=5)
                x = label(x)
                mask[dim, bbox[0]:bbox[2], bbox[1]:bbox[3]], cc = x > 0, x.max()
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
