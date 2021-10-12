import dask
import dask.bag
import dask.dataframe
import numpy
import scipy.stats
from skimage.feature import hog, greycomatrix, greycoprops
from skimage.measure import label, regionprops_table, shannon_entropy
import skimage


def shape_features_meta(nchannels):
    props = [
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
    ]
    out = {}
    for i in range(nchannels):
        out.update({f"{p}_{i}": float for p in props})
    return out


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


def intensity_features_meta(nchannels):
    props = [
        'mean',
        'max',
        'min',
        'var',
        'mad',
        'diff_entropy',
        'skewness',
        'kurtosis'
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

    def channel_features(i):
        
        quartiles = numpy.quantile(img[i], q=(0.25, 0.75))
        return {
            f'mean_{i}': numpy.mean(img[i]), 
            f'max_{i}': numpy.mean(img[i]),
            f'min_{i}': numpy.min(img[i]),
            f'var_{i}': numpy.var(img[i]),
            f'mad_{i}': scipy.stats.median_abs_deviation(img[i]),
            f'diff_entropy_{i}': scipy.stats.differential_entropy(img[i]),
            f'skewness_{i}': scipy.stats.skew(img[i]),
            f'kurtosis_{i}': scipy.stats.kurtosis(img[i]),
            f'lower_quartile_{i}': quartiles[0],
            f'upper_quartile_{i}': quartiles[1]
        }

    img = sample.get('pixels')
    img = numpy.reshape(img, newshape=(img.shape[0], -1))
    features_dict = {}
    for i in range(len(img)):
        features_dict.update(channel_features(i))

    return features_dict


def texture_features_meta(nchannels):
    greycoprops = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    nhog = 36

    out = {}
    for i in range(nchannels):
        for n in range(2):
            for m in range(2):
                out.update({f"glcm_{p}_{n}_{m}_{i}": float for p in greycoprops})
        out.update({f"hog_{j}_{i}": float for j in range(nhog)})
        out[f"shannon_entropy_{i}"] = float
    return out


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
                out[f'glcm_{prop}_{n}_{m}_{i}'] = p

        # put hog features in dictionary
        for j in range(len(hog_features)):
            out.update({f'hog_{j}_{i}': hog_features[j]})

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


def bbox_features_meta():
    return {
        "bbox_minr": float,
        "bbox_minc": float,
        "bbox_maxr": float,
        "bbox_maxc": float
    }


def bbox_features(p):
    return {
        "bbox_minr": p["bbox"][0],
        "bbox_minc": p["bbox"][1],
        "bbox_maxr": p["bbox"][2],
        "bbox_maxc": p["bbox"][3],
    }


def extract_features(*, images: dask.bag.Bag, nchannels: int, types: list):
    """
    Extract features from pixel data

    Args:
        images (dask.bag): bag containing dictionaries of image data

    Returns:
        dask.bag: bag containing dictionaries of image features
    """

    def features_partition(part):
        data = []
        for p in part:
            type_dicts = []
            if "bbox" in types:
                type_dicts.append(bbox_features(p))
            if "shape" in types:
                type_dicts.append(shape_features(p))
            if "intensity" in types:
                type_dicts.append(intensity_features(p))
            if "texture" in types:
                type_dicts.append(texture_features(p))

            out = {"idx": p["idx"]}
            for type_dict in type_dicts:
                out.update(type_dict)
            data.append(out)
        return data

    images = images.map_partitions(features_partition)

    meta = {"idx":int}
    if "bbox" in types:
        meta.update(bbox_features_meta())
    if "shape" in types:
        meta.update(shape_features_meta(nchannels))
    if "intensity" in types:
        meta.update(intensity_features_meta(nchannels))
    if "texture" in types:
        meta.update(texture_features_meta(nchannels))
    images_df = images.to_dataframe(meta=meta)

    return images_df
