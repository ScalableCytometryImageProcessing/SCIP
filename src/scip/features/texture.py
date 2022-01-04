from typing import Any, Mapping, List

import numpy
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import sobel
import skimage


distances = [3, 5]


def _texture_features_meta(channel_names: List[str]) -> Mapping[str, Any]:
    graycoprops = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']

    out = {}
    for i in channel_names:
        for n in distances:
            out.update({f"glcm_mean_{p}_{n}_{i}": float for p in graycoprops})
            out.update({f"glcm_std_{p}_{n}_{i}": float for p in graycoprops})
            out.update({f"combined_glcm_mean_{p}_{n}_{i}": float for p in graycoprops})
            out.update({f"combined_glcm_std_{p}_{n}_{i}": float for p in graycoprops})
        out[f"sobel_mean_{i}"] = float
        out[f"sobel_std_{i}"] = float
        out[f"sobel_max_{i}"] = float
        out[f"sobel_min_{i}"] = float
        out[f"combined_sobel_mean_{i}"] = float
        out[f"combined_sobel_std_{i}"] = float
        out[f"combined_sobel_max_{i}"] = float
        out[f"combined_sobel_min_{i}"] = float
    return out


def texture_features(
    sample: Mapping[str, Any],
    channel_names: List[str],
    maximum_pixel_value: int
):
    """Extracts texture features from image.

    Texture features are computed based on the gray co-occurence level matrix and sobel map.
    From the former, a contrast, dissimilarity, homogeneity, energy, correlation and ASM metric
    is computed. From the latter, mean, standard deviation, maximum and minimum values are computed.

    Features are not computed on background-substracted values (as in
    :func:scip.features.intensity.intensity_features) because all features are computed on
    relative changes of neighboring pixels; a substraction does not influence these values.

    Args:
        sample (Mapping[str, Any]): mapping with pixels, mask and combined mask keys.

    Returns:
        Mapping[str, Any]: extacted features.

    """

    def _row(pixels, i):
        angles = [
            numpy.pi / 4,  # 45 degrees
            3 * numpy.pi / 4,  # 135 degrees
            5 * numpy.pi / 4,  # 225 degrees
            7 * numpy.pi / 4  # 315 degrees
        ]

        int_img = skimage.img_as_int(pixels / maximum_pixel_value)
        bin_edges = numpy.histogram_bin_edges(int_img, bins=15)
        int_img = numpy.digitize(int_img, bins=bin_edges, right=True)
        glcm = graycomatrix(
            int_img,
            distances=distances,
            angles=angles,
            levels=16,
            normed=True,
            symmetric=True
        )

        out = {}
        for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
            v = graycoprops(glcm, prop=prop)
            for d, (mu, std) in enumerate(zip(v.mean(axis=1), v.std(axis=1))):
                out[f'glcm_mean_{prop}_{distances[d]}_{i}'] = mu
                out[f'glcm_std_{prop}_{distances[d]}_{i}'] = std

        s = sobel(pixels)
        out[f"sobel_mean_{i}"] = s.mean()
        out[f"sobel_std_{i}"] = s.std()
        out[f"sobel_max_{i}"] = s.max()
        out[f"sobel_min_{i}"] = s.min()

        return out

    features_dict = {}

    mask_pixels = sample["pixels"] * sample["mask"]
    combined_mask_pixels = sample["pixels"] * sample["combined_mask"][numpy.newaxis, ...]
    for i, name in enumerate(channel_names):

        # compute features on channel specific mask
        if numpy.any(sample["mask"][i]):
            features_dict.update(_row(mask_pixels[i], name))

        # always compute features on combined mask (it can never be empty)
        out = _row(combined_mask_pixels[i], name)
        for k in out.keys():
            features_dict[f"combined_{k}"] = out[k]

    return features_dict
