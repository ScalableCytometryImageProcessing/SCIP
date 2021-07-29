from scip.quality_control import intensity_distribution
import numpy as np
import dask
import dask.bag


def sample_normalization(sample, quantiles, masked_quantiles):

    img = sample.get('pixels')
    masked = sample.get('mask_img')
    single_blob_mask = sample.get('single_blob_mask_img')

    normalized = np.empty(img.shape, dtype=float)
    normalized_masked = np.empty(img.shape, dtype=float)
    normalized_single_masked = np.empty(img.shape, dtype=float)

    channels = img.shape[0]

    lower = quantiles[0]
    upper = quantiles[1]

    masked_lower = masked_quantiles[0]
    masked_upper = masked_quantiles[1]

    for i in range(channels):
        # Normalize
        quantile_norm = (img[i] - lower[i]) / (upper[i] - lower[i])
        quantile_norm_masked = (masked[i] - masked_lower[i]) / \
                               (masked_upper[i] - masked_lower[i])
        quantile_single_masked = (single_blob_mask[i] - masked_lower[i]) / \
                                 (masked_upper[i] - masked_lower[i])

        # # Clip
        normalized[i] = np.clip(quantile_norm, 0, 1)
        normalized_masked[i] = np.clip(quantile_norm_masked, 0, 1)
        normalized_single_masked[i] = np.clip(quantile_single_masked, 0, 1)

    sample = sample.copy()
    sample.update({'pixels_norm': normalized, 'masked_img_norm': normalized_masked,
                   'single_blob_mask_img_norm': normalized_single_masked})

    return sample


def quantile_normalization(images: dask.bag.Bag, lower, upper):

    def normalize_partition(part, quantiles, masked_quantiles):
        return [sample_normalization(p, quantiles, masked_quantiles) for p in part]

    quantiles, masked_quantiles = \
        intensity_distribution.get_distributed_partitioned_quantile(images, lower, upper)

    return images.map_partitions(normalize_partition, quantiles, masked_quantiles)
