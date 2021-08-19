from scip.quality_control import intensity_distribution
import numpy as np
import dask
import dask.bag


def sample_normalization(sample, qq):
    """
    Perform min-max normalization using quantiles on original pixel data,
    masked pixel data and flat masked intensities list

    Args:
        sample (dict): dictionary containing image data and mask data
        qq (tuple): (lower, upper) list of quantiles for every channel
        masked_qq (tuple): (lower, upper) list of quantiles for
                                  every channel of masked images

    Returns:
        dict: dictionary including normalized data
    """

    qq = np.nanmedian(qq, axis=-1)

    img = sample.get('pixels')
    normalized = np.empty(img.shape, dtype=float)

    for i in range(len(img)):
        normalized[i] = (img[i] - qq[i, 0]) / (qq[i, 1] - qq[i, 0])

    sample = sample.copy()
    sample["pixels"] = normalized

    return sample


def quantile_normalization(images: dask.bag.Bag, lower, upper):
    """
    Apply min-max normalization on all images, both on original pixel data and masked pixel data

    Args:
        images (dask.bag): bag of dictionaries containing image data
        lower (float): lower quantile percentage that will be used as
                        minimum in the min-max normalization
        upper (float): upper quantile percentage that will be used as
                        maximum in the min-max normalization
    Returns:
        dask.bag: bag of dictionaries including normalized data
    """

    def normalize_partition(part, quantiles):
        return [sample_normalization(p, quantiles) for p in part]

    quantiles = intensity_distribution.get_distributed_partitioned_quantile(images, lower, upper)
    return images.map_partitions(normalize_partition, quantiles)
