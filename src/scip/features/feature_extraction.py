import dask
import dask.bag
import dask.dataframe
import numpy
import scipy.stats

# Scikit image libraries
from skimage.feature import hog, greycomatrix, greycoprops
from skimage.measure import label, regionprops_table, shannon_entropy
import skimage


def shape_features(sample):
    """
        compute regionpropse
    Args:
        sample (dict): dictionary containing image data

    Returns:
        dict: dictionary including new features

    """

    def channel_features(i):
        label_img = label(img[i])
        props = regionprops_table(
            label_image=label_img,
            properties=(
                "area",
                "convex_area",
                "eccentricity",
                "equivalent_diameter",
                "euler_number",
                "feret_diameter_max",
                "filled_area",
                "inertia_tensor",
                "inertia_tensor_eigvals",
                "major_axis_length",
                "minor_axis_length",
                "moments_hu",
                "orientation",
                "perimeter",
                "perimeter_crofton",
                "solidity"
            )
        )
        return props

    img = sample.get('mask')
    features_dict = {}
    for i in range(len(img)):
        props = channel_features(i)
        props = {f"{k}_{i}": v[0] for k,v in props.items()}
        features_dict.update(props)

    return features_dict 


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

    def channel_features(i):
        return {
            f'mean_{i}': numpy.mean(img[i]), 
            f'max_{i}': numpy.mean(img[i]),
            f'min_{i}': numpy.min(img[i]),
            f'var_{i}': numpy.var(img[i]),
            f'mad_{i}': scipy.stats.median_abs_deviation(img[i]),
            f'diff_entropy_{i}': scipy.stats.differential_entropy(img[i]),
            f'skewness_{i}': scipy.stats.skew(img[i]),
            f'kurtosis_{i}': scipy.stats.kurtosis(img[i])
        }

    img = sample.get('flat')
    features_dict = {}
    for i in range(len(img)):
        features_dict.update(channel_features(i))

    return features_dict


def texture_features(sample):
    """
    Find texture features based normalized largest area masked pixel data:
        - HOG

    Args:
        sample (dict): dictionary containing image data

    Returns:
        dict: dictionary including new texture features

    """

    def texture_features(i, pixels_per_cell):
        hog_features = hog(
            img[i],
            orientations=4,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=(1, 1),
            visualize=False
        )

        distances = [1, 2]
        angles = [0, numpy.pi/2]

        int_img = skimage.img_as_ubyte(img[i])
        glcm = greycomatrix(int_img, distances=distances, angles=angles, levels=256)

        out = {}
        for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
            v = greycoprops(glcm, prop=prop)

            for (n, m), p in numpy.ndenumerate(v):
                out[f'glcm_{i}_{prop}_{n}_{m}'] = p

        # put hog features in dictionary
        for j in range(len(hog_features)):
            out.update({f'hog_ch_{i}_{j}': hog_features[j]})

        out["shannon_entropy_{i}"] = shannon_entropy(img[i])

        return out

    img = sample.get('pixels')

    # the amount of hog features depends on the size of the input image, which is not uniform
    # for most datasets. Therefore, we dynamically compute the HOG parameters so that there is
    # always a 3x3 cell grid leading to a uniform length feature vector
    pixels_per_cell = img.shape[1] // 3, img.shape[2] // 3

    features_dict = {}
    for i in range(len(img)):
        features_dict.update(texture_features(i, pixels_per_cell))

    return features_dict


def bbox_features(p):
    return {
        "bbox_minr": p["bbox"][0],
        "bbox_minc": p["bbox"][1],
        "bbox_maxr": p["bbox"][2],
        "bbox_maxc": p["bbox"][3],
    }


def extract_features(*, images: dask.bag.Bag):
    """
    Extract features from pixel data

    Args:
        images (dask.bag): bag containing dictionaries of image data

    Returns:
        dask.bag: bag containing dictionaries of image features
    """

    def features_partition(part):
        return [{
            "idx": p["idx"],
            **bbox_features(p),
            **shape_features(p),
            **intensity_features(p),
            **texture_features(p)
        } for p in part]

    images = images.map_partitions(features_partition).persist()
    images = images.to_dataframe()
    # setting the index causes partition divisions to be known for Dask
    # making concatenation fast
    images = images.set_index("idx")

    return images
