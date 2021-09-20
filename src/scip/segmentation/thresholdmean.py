import numpy
from skimage.restoration import denoise_wavelet, estimate_sigma
from skimage.filters import sobel, threshold_mean
from skimage.morphology import label, remove_small_objects, binary_closing, disk
from scip.segmentation import mask_apply


def get_mask(el):

    mask = numpy.empty(shape=el["pixels"].shape, dtype=bool)

    for dim in range(len(el["pixels"])):
        img = el["pixels"][dim]

        denoised = denoise_wavelet(
            img, method="VisuShrink", sigma=estimate_sigma(img), rescale_sigma=True)

        elev = sobel(denoised)

        thresh = elev > threshold_mean(elev)
        
        labeled = label(thresh)

        if numpy.max(labeled) > 1:
            labeled = remove_small_objects(labeled, min_size=30)

        if numpy.max(labeled) > 1:
            mask[dim] = False
        else:
            mask[dim] = binary_closing(labeled, selem=disk(2))

    out = el.copy()
    out["intermediate"] = mask

    return out


def create_masks_on_bag(bag, **kwargs):

    def threshold_masking(partition):
        return [get_mask(p) for p in partition]

    bag = bag.map_partitions(threshold_masking)
    bag = bag.map_partitions(mask_apply.apply_mask_partition)

    return dict(threshold=bag)
