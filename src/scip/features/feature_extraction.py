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

    img = sample['mask']

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

    features_dict = {}
    for i in range(len(img)):
        if numpy.any(img[i]):
            props = channel_features(i)
            props = {f"{k}_{i}": numpy.mean(v) for k, v in props.items()}
            features_dict.update(props)
        else:
            # setting proper default values if possible when the mask is empty
            features_dict.update({
                f"area_{i}": 0,
                f"convex_area_{i}": 0,
                f"eccentricity_{i}": None,
                f"equivalent_diameter_{i}": 0,
                f"euler_number_{i}": None,
                f"feret_diameter_max_{i}": 0,
                f"filled_area_{i}": 0,
                f"inertia_tensor_{i}": None,
                f"inertia_tensor_eigvals_{i}": None,
                f"major_axis_length_{i}": 0,
                f"minor_axis_length_{i}": 0,
                f"moments_hu_{i}": None,
                f"orientation_{i}": None,
                f"perimeter_{i}": 0,
                f"perimeter_crofton_{i}": 0,
                f"solidity_{i}": None
            })

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

    def channel_features(i, values):
        if numpy.any(sample["mask"][i]):
            quartiles = numpy.quantile(values, q=(0.25, 0.75))

            d = {
                f'mean_{i}': numpy.mean(values),
                f'max_{i}': numpy.mean(values),
                f'min_{i}': numpy.min(values),
                f'var_{i}': numpy.var(values),
                f'mad_{i}': scipy.stats.median_abs_deviation(values),
                f'skewness_{i}': scipy.stats.skew(values),
                f'kurtosis_{i}': scipy.stats.kurtosis(values),
                f'lower_quartile_{i}': quartiles[0],
                f'upper_quartile_{i}': quartiles[1]
            }

            window_length = int(numpy.floor(numpy.sqrt(values.size) + 0.5))
            if window_length >= values.size // 2:
                window_length = values.size // 2 - 1

            if window_length < 1:
                diff_ent = None
            else:
                diff_ent = scipy.stats.differential_entropy(
                    values, window_length=window_length)
            d[f'diff_entropy_{i}'] = diff_ent
            return d
        else:
            return {
                f'mean_{i}': 0,
                f'max_{i}': 0,
                f'min_{i}': 0,
                f'var_{i}': 0,
                f'mad_{i}': 0,
                f'skewness_{i}': 0,
                f'kurtosis_{i}': 0,
                f'lower_quartile_{i}': 0,
                f'upper_quartile_{i}': 0,
                f'diff_entropy_{i}': 0
            }

    img = sample['pixels']
    features_dict = {}
    for i in range(len(img)):
        values = img[i][sample["mask"][i]]
        features_dict.update(channel_features(i, values))

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
        angles = [0, numpy.pi / 2]

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

    img = sample['pixels']

    # the amount of hog features depends on the size of the input image, which is not uniform
    # for most datasets. Therefore, we dynamically compute the HOG parameters so that there is
    # always a 3x3 cell grid leading to a uniform length feature vector
    pixels_per_cell = img.shape[1] // 3, img.shape[2] // 3

    features_dict = {}
    for i in range(len(img)):
        if numpy.any(sample["mask"][i]):
            features_dict.update(texture_features(i, pixels_per_cell))

    return features_dict


def bbox_features_meta(nchannels):
    d = {
        "bbox_minr": float,
        "bbox_minc": float,
        "bbox_maxr": float,
        "bbox_maxc": float
    }
    d.update({f"regions_{i}": int for i in range(nchannels)})
    return d


def bbox_features(p):
    d = {
        "bbox_minr": p["bbox"][0],
        "bbox_minc": p["bbox"][1],
        "bbox_maxr": p["bbox"][2],
        "bbox_maxc": p["bbox"][3],
    }
    d.update({f"regions_{i}": c for i, c in enumerate(p["regions"])})
    return d


def extract_features(*, images: dask.bag.Bag, nchannels: int, types: list):  # noqa: C901
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

    meta = {"idx": str}
    if "bbox" in types:
        meta.update(bbox_features_meta(nchannels))
    if "shape" in types:
        meta.update(shape_features_meta(nchannels))
    if "intensity" in types:
        meta.update(intensity_features_meta(nchannels))
    if "texture" in types:
        meta.update(texture_features_meta(nchannels))
    images_df = images.to_dataframe(meta=meta)

    return images_df
