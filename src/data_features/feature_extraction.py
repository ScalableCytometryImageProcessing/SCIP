import dask
import dask.bag
import numpy as np

# Scikit image libraries
from skimage.feature import hog
from skimage.measure import label, regionprops


def shape_features(sample):

    # def largest_region(regions):
    #     largest = 0
    #     largest_index = 0
    #     for props in regions:
    #         if props.area > largest:
    #             largest = props.area
    #             largest_index = regions.index(props)
    #     return largest_index

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
    entries_to_remove = ('pixels', 'denoised', 'segmented', 'mask', 'mask_img', 'single_blob_mask',
                         'pixels_norm', 'masked_img_norm', 'single_blob_mask_img_norm', 'single_blob_mask_img')
    for k in entries_to_remove:
        sample.pop(k, None)
    
    return sample


def extract_features(images: dask.bag.Bag):

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
