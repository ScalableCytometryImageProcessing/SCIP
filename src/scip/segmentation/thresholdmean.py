import numpy
from skimage import morphology
from skimage.restoration import denoise_wavelet, estimate_sigma
from skimage.filters import sobel, threshold_mean
from scip.segmentation import util


def get_mask(el):

    mask = numpy.empty(shape=el["pixels"].shape, dtype=bool)
    connected_components = []

    for dim in range(len(el["pixels"])):
        img = el["pixels"][dim]

        denoised = denoise_wavelet(
            img, method="VisuShrink", sigma=estimate_sigma(img), rescale_sigma=True)

        elev = sobel(denoised)
        closed = morphology.closing(elev, selem=morphology.disk(2))
        thresh = closed > threshold_mean(closed)

        mask[dim], cc = util.mask_post_process(thresh)
        connected_components.append(cc)

    out = el.copy()
    out["intermediate"] = mask
    out["connected_components"] = connected_components

    return out


def create_masks_on_bag(bag, **kwargs):

    def threshold_masking(partition):
        return [get_mask(p) for p in partition]

    bag = bag.map_partitions(threshold_masking)
    bag = bag.map_partitions(util.apply_mask_partition)

    return dict(threshold=bag)
