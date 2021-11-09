import numpy
from skimage.feature import hog, greycomatrix, greycoprops
from skimage.measure import shannon_entropy
import skimage


distances = [3, 5]


def texture_features_meta(nchannels):
    greycoprops = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    nhog = 36

    out = {}
    for i in range(nchannels):
        for n in distances:
            out.update({f"glcm_{p}_{n}_{i}": float for p in greycoprops})
            out.update({f"bgcorr_glcm_{p}_{n}_{i}": float for p in greycoprops})
        out.update({f"hog_{j}_{i}": float for j in range(nhog)})
        out.update({f"bgcorr_hog_{j}_{i}": float for j in range(nhog)})
        out[f"shannon_entropy_{i}"] = float
        out[f"bgcorr_shannon_entropy_{i}"] = float
    return out


def texture_features(sample, maximum_pixel_value):
    """

    Args:
        sample (dict): dictionary containing image data

    Returns:
        dict: dictionary including new texture features

    """

    def row(pixels, pixels_per_cell, i):
        hog_features = hog(
            pixels,
            orientations=4,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=(1, 1),
            visualize=False
        )

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

            for d, p in enumerate(v.mean(axis=1)):
                out[f'glcm_{prop}_{distances[d]}_{i}'] = p

        # put hog features in dictionary
        for j in range(len(hog_features)):
            out.update({f'hog_{j}_{i}': hog_features[j]})

        out["shannon_entropy_{i}"] = shannon_entropy(pixels+1)

        return out

    # the amount of hog features depends on the size of the input image, which is not uniform
    # for most datasets. Therefore, we dynamically compute the HOG parameters so that there is
    # always a 3x3 cell grid leading to a uniform length feature vector
    pixels_per_cell = sample["pixels"].shape[1] // 3, sample["pixels"].shape[2] // 3

    features_dict = {}
    for i in range(len(sample["pixels"])):
        if numpy.any(sample["mask"][i]):
            features_dict.update(
                row(
                    sample["pixels"][i],
                    pixels_per_cell,
                    i
                )
            )
            bg_sub = sample["pixels"][i].copy()
            bg_sub[sample["mask"][i]] -= sample["mean_background"][i]
            tmp = row(bg_sub, pixels_per_cell, i)
            tmp2 = {}
            for k in tmp.keys():
                tmp2[f"bgcorr_{k}"] = tmp[k]
            features_dict.update(tmp2)

    return features_dict
