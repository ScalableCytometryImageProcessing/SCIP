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

    img = sample.get('single_blob_mask')
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

    img = sample.get('masked_img_norm')
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

    def texture_features(i):
        channel_img = img[i]
        hog_features = hog(channel_img, orientations=8, pixels_per_cell=(16, 16),
                           cells_per_block=(1, 1))
        hog_dict = {}

        # If no hog features can be found inser NaN
        if len(hog_features) == 0:
            for j in range(48):
                hog_dict.update({f'hog_ch_{i}_{j}': np.NaN})
            return hog_dict

        # put hog features in dictionary
        for j in range(48):
            hog_dict.update({f'hog_ch_{i}_{j}': hog_features[j]})

        return hog_dict

    img = sample.get('single_blob_mask_img_norm')
    channels = img.shape[0]

    features_dict = {"idx": sample["idx"]}
    for i in range(channels):
        features_dict.update(texture_features(i))

    return features_dict


def extract_features(images: dask.bag.Bag):
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

    shape_df = images.map_partitions(shape_partition).to_dataframe().set_index("idx")
    intensity_df = images.map_partitions(intensity_partition).to_dataframe().set_index("idx")
    texture_df = images.map_partitions(texture_partition).to_dataframe().set_index("idx")

    return dask.dataframe.multi.concat([shape_df, intensity_df, texture_df], axis=1)
