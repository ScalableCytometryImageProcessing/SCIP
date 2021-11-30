import numpy
from skimage.feature import greycomatrix, greycoprops
from skimage.measure import shannon_entropy
from skimage.filters import sobel
import skimage


distances = [3, 5]


def texture_features_meta(nchannels):
    greycoprops = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']

    out = {}
    for i in range(nchannels):
        for n in distances:
            out.update({f"glcm_{p}_{n}_{i}": float for p in greycoprops})
            out.update({f"bgcorr_glcm_{p}_{n}_{i}": float for p in greycoprops})
        out[f"shannon_entropy_{i}"] = float
        out[f"bgcorr_shannon_entropy_{i}"] = float
        out[f"sobel_mean_{i}"] = float
        out[f"sobel_std_{i}"] = float
        out[f"sobel_max_{i}"] = float
        out[f"sobel_min_{i}"] = float
    return out


def texture_features(sample, maximum_pixel_value):
    """

    Args:
        sample (dict): dictionary containing image data

    Returns:
        dict: dictionary including new texture features

    """

    def row(pixels, i, bg_subbed=False):
        angles = [
            numpy.pi / 4,  # 45 degrees
            3 * numpy.pi / 4,  # 135 degrees
            5 * numpy.pi / 4,  # 225 degrees
            7 * numpy.pi / 4  # 315 degrees
        ]

        int_img = skimage.img_as_int(pixels / maximum_pixel_value)
        bin_edges = numpy.histogram_bin_edges(int_img, bins=15)
        int_img = numpy.digitize(int_img, bins=bin_edges, right=True)
        glcm = greycomatrix(
            int_img,
            distances=distances,
            angles=angles,
            levels=16,
            normed=True,
            symmetric=True
        )

        out = {}
        for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
            v = greycoprops(glcm, prop=prop)
            for d, (mu, std) in enumerate(zip(v.mean(axis=1), v.std(axis=1))):
                out[f'glcm_mean_{prop}_{distances[d]}_{i}'] = mu
                out[f'glcm_std_{prop}_{distances[d]}_{i}'] = std

        out["shannon_entropy_{i}"] = shannon_entropy(int_img)

        if not bg_subbed:
            s = sobel(pixels)
            out[f"sobel_mean_{i}"] = s.mean()
            out[f"sobel_std_{i}"] = s.std()
            out[f"sobel_max_{i}"] = s.max()
            out[f"sobel_min_{i}"] = s.min()

        return out

    features_dict = {}
    for i in range(len(sample["pixels"])):
        if numpy.any(sample["mask"][i]):
            features_dict.update(row(sample["pixels"][i], i))
            bg_sub = sample["pixels"][i].copy()
            bg_sub[sample["mask"][i]] -= sample["mean_background"][i]
            for k, v in row(bg_sub, i, bg_subbed=True).items():
                features_dict[f"bgcorr_{k}"] = v

    return features_dict
