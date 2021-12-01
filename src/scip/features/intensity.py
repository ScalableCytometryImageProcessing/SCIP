import numpy
import scipy.stats
from scipy.ndimage import convolve

props = [
    'mean',
    'max',
    'min',
    'var',
    'mad',
    'skewness',
    'kurtosis',
    'sum',
    'modulation'
]


def intensity_features_meta(nchannels):
    out = {}
    for i in range(nchannels):
        out.update({f"{p}_{i}": float for p in props})
        out.update({f"bgcorr_{p}_{i}": float for p in props})
        out.update({f"edge_{p}_{i}": float for p in props})
        out.update({f"bgcorr_edge_{p}_{i}": float for p in props})
        out.update({f"combined_{p}_{i}": float for p in props})
        out.update({f"combined_bgcorr_{p}_{i}": float for p in props})
        out.update({f"combined_edge_{p}_{i}": float for p in props})
        out.update({f"combined_bgcorr_edge_{p}_{i}": float for p in props})
    return out


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

    return d


def intensity_features(sample):
    """
    Find following intensity features based on normalized masked pixel data:
        'mean',
        'max',
        'min',
        'var',
        'mad',
        'skewness',
        'kurtosis',
        'sum',
        'modulation'
    The features are computed on 8 different views on the pixel data:
        1. Raw values of channel specific mask
        2. Background substracted values of channel specific mask
        3. Edge values of channel specific mask
        4. Background substracted edge values of channel specific mask
        5-8: Views 1-4 but on union of masks of all channels

    Args:
        sample (dict): dictionary including image data

    Returns:
        dict: dictionary including computed intensity features
    """

    features_dict = {}

    conv = convolve(
        sample["combined_mask"],
        weights=numpy.ones(shape=[3, 3], dtype=int),
        mode="constant"
    )
    combined_edge = (conv > 0) & (conv < 9)

    for i in range(len(sample["pixels"])):

        # compute features on channel specific mask
        if numpy.any(sample["mask"][i]):

            mask_pixels = sample["pixels"][i][sample["mask"][i]]
            mask_bgcorr_pixels = mask_pixels - sample["background"][i]

            conv = convolve(
                sample["mask"][i],
                weights=numpy.ones(shape=[3, 3], dtype=int),
                mode="constant"
            )
            edge = (conv > 0) & (conv < 9)
            mask_edge_pixels = sample["pixels"][i][edge]
            mask_bgcorr_edge_pixels = mask_edge_pixels - sample["background"][i]

            mask_out = row(mask_pixels, i)
            mask_bgcorr_out = row(mask_bgcorr_pixels, i)
            mask_edge_out = row(mask_edge_pixels, i)
            mask_bgcorr_edge_out = row(mask_bgcorr_edge_pixels, i)

            for k in mask_out.keys():
                features_dict[k] = mask_out[k]
                features_dict[f"bgcorr_{k}"] =  mask_bgcorr_out[k]
                features_dict[f"edge_{k}"] = mask_edge_out[k]
                features_dict[f"bgcorr_edge_{k}"] = mask_bgcorr_edge_out[k]
        else:
            # write default values
            for k in props:
                features_dict[k] = 0
                features_dict[f"bgcorr_{k}"] = 0
                features_dict[f"edge_{k}"] = 0
                features_dict[f"bgcorr_edge_{k}"] = 0

        # always compute features on combined mask (it can never be empty)
        mask_pixels = sample["pixels"][i][sample["combined_mask"]]
        mask_bgcorr_pixels = mask_pixels - sample["combined_background"][i]

        mask_edge_pixels = sample["pixels"][i][combined_edge]
        mask_bgcorr_edge_pixels = mask_edge_pixels - sample["combined_background"][i]

        mask_out = row(mask_pixels, i)
        mask_bgcorr_out = row(mask_bgcorr_pixels, i)
        mask_edge_out = row(mask_edge_pixels, i)
        mask_bgcorr_edge_out = row(mask_bgcorr_edge_pixels, i)

        for k in mask_out.keys():
            features_dict[f"combined_{k}"] = mask_out[k]
            features_dict[f"combined_bgcorr_{k}"] =  mask_bgcorr_out[k]
            features_dict[f"combined_edge_{k}"] = mask_edge_out[k]
            features_dict[f"combined_bgcorr_edge_{k}"] = mask_bgcorr_edge_out[k]

    return features_dict
