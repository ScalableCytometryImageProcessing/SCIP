import numpy
import scipy.stats
from scipy.ndimage import convolve


def intensity_features_meta(nchannels):
    props = [
        'mean',
        'max',
        'min',
        'var',
        'mad',
        'skewness',
        'kurtosis',
        'sum',
        'modulation',
        'edge_mean',
        'edge_max',
        'edge_min',
        'edge_var',
        'edge_mad',
        'edge_skewness',
        'edge_kurtosis',
        'edge_sum',
        'edge_modulation',
        'bgcorr_mean',
        'bgcorr_max',
        'bgcorr_min',
        'bgcorr_var',
        'bgcorr_mad',
        'bgcorr_skewness',
        'bgcorr_kurtosis',
        'bgcorr_sum',
        'bgcorr_modulation',
        'bgcorr_edge_mean',
        'bgcorr_edge_max',
        'bgcorr_edge_min',
        'bgcorr_edge_var',
        'bgcorr_edge_mad',
        'bgcorr_edge_skewness',
        'bgcorr_edge_kurtosis',
        'bgcorr_edge_sum',
        'bgcorr_edge_modulation'
    ]
    out = {}
    for i in range(nchannels):
        out.update({f"{p}_{i}": float for p in props})
    return out


def intensity_features(sample):
    """
    Find following intensity features based on normalized masked pixel data:
        - mean
        - max
        - min

    Args:
        sample (dict): dictionary including image data

    Returns:
        dict: dictionary including new intensity features
    """

    def row(pixels, i):
        quartiles = numpy.quantile(pixels, q=(0.25, 0.75))

        d = {
            f'mean_{i}': numpy.mean(pixels),
            f'max_{i}': numpy.mean(pixels),
            f'min_{i}': numpy.min(pixels),
            f'var_{i}': numpy.var(pixels),
            f'mad_{i}': scipy.stats.median_abs_deviation(pixels),
            f'skewness_{i}': scipy.stats.skew(pixels),
            f'kurtosis_{i}': scipy.stats.kurtosis(pixels),
            f'lower_quartile_{i}': quartiles[0],
            f'upper_quartile_{i}': quartiles[1],
            f'sum_{i}': numpy.sum(pixels)
        }
        d[f"modulation_{i}"] = (d[f"max_{i}"] - d[f"min_{i}"]) / ((d[f"max_{i}"] + d[f"min_{i}"]))

        # compute features only on edge pixels
        conv = convolve(
            sample["mask"][i],
            weights=numpy.ones(shape=[3, 3], dtype=int),
            mode="constant"
        )
        edge = (conv > 0) & (conv < 9)
        pixels = sample["pixels"][i][edge]

        quartiles = numpy.quantile(pixels, q=(0.25, 0.75))

        d.update({
            f'edge_mean_{i}': numpy.mean(pixels),
            f'edge_max_{i}': numpy.mean(pixels),
            f'edge_min_{i}': numpy.min(pixels),
            f'edge_var_{i}': numpy.var(pixels),
            f'edge_mad_{i}': scipy.stats.median_abs_deviation(pixels),
            f'edge_skewness_{i}': scipy.stats.skew(pixels),
            f'edge_kurtosis_{i}': scipy.stats.kurtosis(pixels),
            f'edge_lower_quartile_{i}': quartiles[0],
            f'edge_upper_quartile_{i}': quartiles[1],
            f'edge_sum_{i}': numpy.sum(pixels)
        })
        d[f"edge_modulation_{i}"] = (
            d[f"edge_max_{i}"] - d[f"edge_min_{i}"]) / ((d[f"edge_max_{i}"] + d[f"edge_min_{i}"]))

        return d

    features_dict = {}
    for i in range(len(sample["pixels"])):
        if numpy.any(sample["mask"][i]):
            features_dict.update(row(sample["pixels"][i][sample["mask"][i]], i))
            bg_sub = sample["pixels"][i][sample["mask"][i]] - sample["mean_background"][i]

            for k, v in row(bg_sub, i).items():
                features_dict[f"bgcorr_{k}"] = v
        else:
            features_dict.update({
                f'mean_{i}': 0,
                f'max_{i}': 0,
                f'min_{i}': 0,
                f'var_{i}': 0,
                f'mad_{i}': 0,
                f'skewness_{i}': 0,
                f'kurtosis_{i}': 0,
                f'lower_quartile_{i}': 0,
                f'upper_quartile_{i}': 0,
                f'diff_entropy_{i}': 0,
                f'sum_{i}': 0,
                f'edge_mean_{i}': 0,
                f'edge_max_{i}': 0,
                f'edge_min_{i}': 0,
                f'edge_var_{i}': 0,
                f'edge_mad_{i}': 0,
                f'edge_skewness_{i}': 0,
                f'edge_kurtosis_{i}': 0,
                f'edge_lower_quartile_{i}': 0,
                f'edge_upper_quartile_{i}': 0,
                f'edge_diff_entropy_{i}': 0,
                f'edge_sum_{i}': 0,
                f'bgcorr_mean_{i}': 0,
                f'bgcorr_max_{i}': 0,
                f'bgcorr_min_{i}': 0,
                f'bgcorr_var_{i}': 0,
                f'bgcorr_mad_{i}': 0,
                f'bgcorr_skewness_{i}': 0,
                f'bgcorr_kurtosis_{i}': 0,
                f'bgcorr_lower_quartile_{i}': 0,
                f'bgcorr_upper_quartile_{i}': 0,
                f'bgcorr_diff_entropy_{i}': 0,
                f'bgcorr_sum_{i}': 0,
                f'bgcorr_edge_mean_{i}': 0,
                f'bgcorr_edge_max_{i}': 0,
                f'bgcorr_edge_min_{i}': 0,
                f'bgcorr_edge_var_{i}': 0,
                f'bgcorr_edge_mad_{i}': 0,
                f'bgcorr_edge_skewness_{i}': 0,
                f'bgcorr_edge_kurtosis_{i}': 0,
                f'bgcorr_edge_lower_quartile_{i}': 0,
                f'bgcorr_edge_upper_quartile_{i}': 0,
                f'bgcorr_edge_diff_entropy_{i}': 0,
                f'bgcorr_edge_sum_{i}': 0
            })

    return features_dict
