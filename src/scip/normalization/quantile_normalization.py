import numpy as np
import dask
import dask.bag


def get_distributed_partitioned_quantile(bag, lower, upper):
    """
    Third method for quantile calculation:
    In every partition intensities are grouped together per channel, on this grouping
    a quantile calculation is performed. The found quantiles per partition are then reduced
    with a median.
    """

    def select_origin(partition, *, origin):
        """
        Maps each element in the partition to the requested and flattened origin values
        """

        mapped = []
        for el in partition:

            # values cannot be a numpy array as not all channels are required to have the same
            # amount of values for one element (due to masking)
            values = []
            for v in el[origin]:
                values.append(v.flatten())
            mapped.append(values)

        return mapped

    def concatenate_lists(a, b):
        """
        Concatenates the numpy vectors in list a and b element-wise
        """
        for i in range(len(b)):
            a[i] = np.concatenate((a[i], b[i]))

        return a

    def reduce_quantiles(a, b):
        """
        Reduces numpy vectors in lists a and b to their quantiles and concatenates them
        """

        if not hasattr(a, "shape"):
            a = np.array([np.quantile(v, (lower, upper)) for v in a])[..., np.newaxis]
        b = np.array([np.quantile(v, (lower, upper)) for v in b])[..., np.newaxis]
        return np.concatenate([a, b], axis=-1)

    qq = bag.map_partitions(select_origin, origin="flat")
    qq = qq.fold(concatenate_lists, reduce_quantiles)

    def quantiles(a):
        out = np.empty(shape=(len(a), 2))
        out[:, 0] = np.min(a[:, 0], axis=-1)
        out[:, 1] = np.max(a[:, 1], axis=-1)
        return out

    qq = qq.apply(quantiles)

    return qq


def sample_normalization(sample, qq):
    """
    Perform min-max normalization using quantiles on original pixel data,
    masked pixel data and flat masked intensities list

    Args:
        sample (dict): dictionary containing image data and mask data
        qq: (lower, upper) list of quantiles for every channel
    Returns:
        dict: dictionary including normalized data
    """

    sample = sample.copy()
    for i in range(len(sample["pixels"])):
        sample["flat"][i] = np.clip((sample["flat"][i] - qq[i, 0]) / (qq[i, 1] - qq[i, 0]), 0, 1)
        sample["pixels"][i][sample["mask"][i]] = sample["flat"][i]

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

    quantiles = get_distributed_partitioned_quantile(images, lower, upper)
    return images.map_partitions(normalize_partition, quantiles)
