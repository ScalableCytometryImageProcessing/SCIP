import dask
import dask.bag
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
            return {f'minor_axis_{i}': float('NaN'),
                    f'major_axis_length_{i}': float('NaN'),
                    f'area_{i}': float('NaN'),
                    f'perimeter_{i}': float('NaN'),
                    f'solidity_{i}': float('NaN'),
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
    features_dict = {}
    for i in range(channels):
        features_dict.update(channel_features(i))

    return {**sample, **features_dict}


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
    features_dict = {}
    for i in range(channels):
        features_dict.update(channel_features(i))

    return {**sample, **features_dict}


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
            hog_dict.update({f'hog_ch_{i}_{j}': hog_features[i]})

        return hog_dict

    img = sample.get('single_blob_mask_img_norm')
    channels = img.shape[0]

    features_dict = {}
    for i in range(channels):
        features_dict.update(texture_features(i))

    return {**sample, **features_dict}


def remove_keys(sample):
    """
    After feature extraction remove unnecessary pixel data

    Args:
        sample (dict): dictionary containing features and pixel data

    Returns:
        dict: dictionary only containing features and path
    """
    entries_to_remove = ('pixels', 'denoised', 'segmented', 'mask',
                         'mask_img', 'single_blob_mask', 'pixels_norm',
                         'masked_img_norm', 'single_blob_mask_img_norm',
                         'single_blob_mask_img')
    for k in entries_to_remove:
        sample.pop(k, None)

    return sample


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

    def remove_redundant_keys(part):
        return [remove_keys(p) for p in part]

    return (
        images
        .map_partitions(shape_partition)
        .map_partitions(intensity_partition)
        .map_partitions(texture_partition)
        .map_partitions(remove_redundant_keys)
        .to_dataframe()
    )
