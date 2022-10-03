import numpy
from skimage.filters.rank import median
from skimage.filters import gaussian, threshold_otsu, sobel
from skimage.morphology import disk, remove_small_holes, remove_small_objects, label
from scip.utils.util import check


@check
def get_mask(el):

    regions = [0] * len(el["pixels"])
    mask = numpy.full(shape=el["pixels"].shape, dtype=bool, fill_value=False)

    # load over channels, starting with main_channel
    for dim in numpy.arange(len(el["pixels"])):
        cc = 0

        if el["mask_filter"][dim]:

            x = el["pixels"][dim]
            if x.max() > 512:
                x = numpy.digitize(
                    x,
                    bins=numpy.histogram_bin_edges(x.ravel(), bins=512)
                ).astype('uint16')

            p = x.copy()
            p = median(p, footprint=disk(5))
            p = x.astype(float) - p
            p = gaussian(p, sigma=.5)
            p = sobel(p)

            p = (p - p.min()) / (p.max() - p.min())
            p = (p * x.max()).astype('uint16')

            p = median(p, footprint=disk(5))

            p = p > threshold_otsu(p)

            p = remove_small_holes(p, area_threshold=(p.shape[0] * p.shape[1]) / 4)
            p = remove_small_objects(p, min_size=20)
            p = label(p)

            mask[dim], cc = p > 0, p.max()

        regions[dim] = cc

    out = el.copy()
    out["mask"] = mask
    out["regions"] = regions

    return out


def create_masks_on_bag(bag):

    def masking(partition):
        return [get_mask(p) for p in partition]

    bag = bag.map_partitions(masking)
    return bag
