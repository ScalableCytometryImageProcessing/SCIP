import numpy as np
import dask
import dask.bag


def get_distributed_minmax(bag, nchannels):  # noqa: C901

    def combine_extent_partition(a, b):

        b = b["pixels"]

        out = np.empty(shape=a.shape)
        for i in range(len(b)):
            if b[i].size == 0:
                out[i] = a[i]
            else:
                out[i, 0] = min(a[i, 0], np.min(b[i]))
                out[i, 1] = max(a[i, 1], np.max(b[i]))
        return out

    def final_minmax(a, b):
        out = np.empty(shape=a.shape)
        for i in range(len(a)):
            out[i, 0] = min(a[i, 0], b[i, 0])
            out[i, 1] = max(a[i, 1], b[i, 1])
        return out

    init = np.empty(shape=(nchannels, 2))
    init[:, 0] = np.inf
    init[:, 1] = -np.inf
    out = bag.foldby(
        key="groupidx",
        binop=combine_extent_partition, 
        combine=final_minmax, 
        initial=init
    )

    return out


def get_distributed_partitioned_quantile(bag, lower, upper):
    """
    Third method for quantile calculation:
    In every partition intensities are grouped together per channel, on this grouping
    a quantile calculation is performed. The found quantiles per partition are then reduced
    with a median.
    """

    def concatenate_lists(a, b):
        """
        Concatenates the numpy vectors in list a and b element-wise
        """
        for i in range(len(b)):
            a[i] = np.concatenate((a[i].flatten(), b[i].flatten()))

        return a

    def reduce_quantiles(a, b):
        """
        Reduces numpy vectors in lists a and b to their quantiles and concatenates them
        """

        if not hasattr(a, "shape"):
            a = np.array([np.quantile(v, (lower, upper)) for v in a])[..., np.newaxis]
        b = np.array([np.quantile(v, (lower, upper)) for v in b])[..., np.newaxis]
        return np.concatenate([a, b], axis=-1)

    qq = bag.fold(concatenate_lists, reduce_quantiles)

    def quantiles(a):
        out = np.empty(shape=(len(a), 2))
        out[:, 0] = np.min(a[:, 0], axis=-1)
        out[:, 1] = np.max(a[:, 1], axis=-1)
        return out

    qq = qq.apply(quantiles)

    return qq


def sample_normalization(sample, quantiles):
    """
    Perform min-max normalization using quantiles on original pixel data,
    masked pixel data and flat masked intensities list

    Args:
        sample (dict): dictionary containing image data and mask data
        qq: (lower, upper) list of quantiles for every channel
    Returns:
        dict: dictionary including normalized data
    """

    qq = dict(quantiles)[sample["groupidx"]]

    sample = sample.copy()
    for i in range(len(sample["pixels"])):
        flat = sample["pixels"][i][sample["mask"][i]]
        sample["pixels"][i][sample["mask"][i]] = np.clip(
            (flat - qq[i, 0]) / (qq[i, 1] - qq[i, 0]), 0, 1)
    return sample


def quantile_normalization(images: dask.bag.Bag, lower, upper, nchannels):
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

    if lower == 0 and upper == 1:
        quantiles = get_distributed_minmax(images, nchannels)
    else:
        quantiles = get_distributed_partitioned_quantile(images, lower, upper)

    images = images.map_partitions(
        normalize_partition, quantiles.to_delayed()[0])

    return images
