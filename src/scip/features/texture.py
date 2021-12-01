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
            out.update({f"glcm_mean_{p}_{n}_{i}": float for p in greycoprops})
            out.update({f"glcm_std_{p}_{n}_{i}": float for p in greycoprops})
            out.update({f"combined_glcm_mean_{p}_{n}_{i}": float for p in greycoprops})
            out.update({f"combined_glcm_std_{p}_{n}_{i}": float for p in greycoprops})
        out[f"sobel_mean_{i}"] = float
        out[f"sobel_std_{i}"] = float
        out[f"sobel_max_{i}"] = float
        out[f"sobel_min_{i}"] = float
        out[f"combined_sobel_mean_{i}"] = float
        out[f"combined_sobel_std_{i}"] = float
        out[f"combined_sobel_max_{i}"] = float
        out[f"combined_sobel_min_{i}"] = float
    return out


def texture_features(sample, maximum_pixel_value):
    """

    Args:
        sample (dict): dictionary containing image data

    Returns:
        dict: dictionary including new texture features

    """

    def row(pixels, i):
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

        s = sobel(pixels)
        out[f"sobel_mean_{i}"] = s.mean()
        out[f"sobel_std_{i}"] = s.std()
        out[f"sobel_max_{i}"] = s.max()
        out[f"sobel_min_{i}"] = s.min()

        return out

    features_dict = {}

    mask_pixels = sample["pixels"] * sample["mask"]
    combined_mask_pixels = sample["pixels"] * sample["combined_mask"][numpy.newaxis, ...]
    for i in range(len(sample["pixels"])):

        # compute features on channel specific mask
        if numpy.any(sample["mask"][i]):
            features_dict.update(row(mask_pixels[i], i))

        # always compute features on combined mask (it can never be empty)
        out = row(combined_mask_pixels[i], i)
        for k in out.keys():
            features_dict[f"combined_{k}"] = out[k]

    return features_dict
