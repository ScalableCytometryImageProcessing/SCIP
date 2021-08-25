import dask
import dask.bag
import dask.dataframe
import numpy as np

# Scikit image libraries
from skimage.feature import hog
from skimage.measure import label, regionprops


def shape_features(sample):
    """
    Find following shape features based on the larges area mask:
        - minor_axis
        - major_axis
        - area
        - perimeter
        - solidity
        - regions

    Args:
        sample (dict): dictionary containing image data

    Returns:
        dict: dictionary including new features

    """

    def channel_features(i):
        label_img = label(img[i])
        regions = regionprops(label_img)
        if len(regions) == 0:
            return {f'minor_axis_{i}': 0.0,
                    f'major_axis_length_{i}': 0.0,
                    f'area_{i}': 0.0,
                    f'perimeter_{i}': 0.0,
                    f'solidity_{i}': 0.0,
                    f'regions_{i}': 0}

        main_part = regions[0]
        return {f'minor_axis_{i}': main_part.minor_axis_length,
                f'major_axis_length_{i}': main_part.major_axis_length,
                f'area_{i}': float(main_part.area),
                f'perimeter_{i}': main_part.perimeter,
                f'solidity_{i}': main_part.solidity,
                f'regions_{i}': len(regions)}

    img = sample.get('pixels')
    channels = img.shape[0]
    features_dict = {"idx": sample["idx"]}
    for i in range(channels):
        features_dict.update(channel_features(i))

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
        channel_img = img[i]
        return {f'mean_{i}': np.mean(channel_img), f'max_{i}': np.mean(channel_img),
                f'min_{i}': np.min(channel_img)}

    img = sample.get('pixels')
    channels = img.shape[0]
    features_dict = {"idx": sample["idx"]}
    for i in range(channels):
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
        channel_img = img[i]
        hog_features = hog(
            channel_img,
            orientations=4,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=(1, 1),
            visualize=False
        )

        hog_dict = {}

        # put hog features in dictionary
        for j in range(len(hog_features)):
            hog_dict.update({f'hog_ch_{i}_{j}': hog_features[j]})

        return hog_dict

    img = sample.get('pixels')

    # the amount of hog features depends on the size of the input image, which is not uniform
    # for most datasets. Therefore, we dynamically compute the HOG parameters so that there is
    # always a 3x3 cell grid leading to a uniform length feature vector 
    pixels_per_cell = img.shape[1] // 3, img.shape[2] // 3

    features_dict = {"idx": sample["idx"]}
    for i in range(len(img)):
        features_dict.update(texture_features(i, pixels_per_cell))

    return features_dict


def extract_features(*, images: dask.bag.Bag):
    """
    Extract features from pixel data

    Args:
        images (dask.bag): bag containing dictionaries of image data

    Returns:
        dask.bag: bag containing dictionaries of image features
    """

    def shape_partition(part):
        return [shape_features(p) for p in part]

    def intensity_partition(part):
        return [intensity_features(p) for p in part]

    def texture_partition(part):
        return [texture_features(p) for p in part]

    def meta_partition(part):
        return [dict(path=p["path"], idx=p["idx"]) for p in part]

    def to_dataframe(bag, prefix):
        df = bag.to_dataframe()

        # setting the index causes partition divisions to be known for Dask
        # making concatenation fast
        df = df.set_index("idx")
        df = df.rename(columns=lambda n: f"{prefix}_{n}")
        return df

    shape_df = to_dataframe(images.map_partitions(shape_partition), "feat")
    intensity_df = to_dataframe(images.map_partitions(intensity_partition), "feat")
    texture_df = to_dataframe(images.map_partitions(texture_partition), "feat")
    meta_df = to_dataframe(images.map_partitions(meta_partition), "meta")

    return dask.dataframe.multi.concat(
        [shape_df, intensity_df, texture_df, meta_df], axis=1)
