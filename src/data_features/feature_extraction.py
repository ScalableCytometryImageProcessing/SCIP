import dask
import dask.bag
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Scikit image libraries
from skimage.draw import ellipse
from skimage.feature import hog
from skimage.measure import label, regionprops, regionprops_table
from skimage.transform import rotate


def shape_features(sample):

    def largest_region(regions):
        largest = 0
        largest_index = 0
        for props in regions:
            if props.area > largest:
                largest = props.area
                largest_index = regions.index(props)
        return largest_index
        

    def channel_features(i):
        label_img = label(img[i])
        regions = regionprops(label_img)
        main_shape = 0
        if len(regions) == 0:
            return {'minor_axis': np.NaN, 
                'major_axis_length': np.NaN, 
                'area': np.NaN, 
                'perimeter': np.NaN, 
                'solidity': np.NaN,
                'regions': 0}   

        if len(regions) > 1:
            main_shape = largest_region(regions)
        main_part = regions[main_shape]     
        return {'minor_axis': main_part.minor_axis_length, 
                'major_axis_length': main_part.major_axis_length, 
                'area': main_part.area, 
                'perimeter': main_part.perimeter, 
                'solidity': main_part.solidity,
                'regions': len(regions)}   


    img = sample.get('mask')
    channels = img.shape[0]
    features = [channel_features(i) for i in range(channels)]

    return {**sample, **dict(shape_features=features)}


def intensity_features(sample):

    def channel_features(i):
        channel_img = img[i]
        return {'mean': np.mean(channel_img), 'max': np.mean(channel_img), 'min': np.min(channel_img)}

    img = sample.get('masked_img_norm')
    channels = img.shape[0]
    features = [channel_features(i) for i in range(channels)]

    return {**sample, **dict(intensity_features=features)}


def texture_features(sample):

    def texture_features(i):
        channel_img = img[i]
        hog_features = hog(channel_img, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1))
        return {'hog': hog_features}

    img = sample.get('masked_img_norm')
    channels = img.shape[0]
    features = [texture_features(i) for i in range(channels)]

    return {**sample, **dict(texture_features=features)}


def extract_features(images: dask.bag.Bag):

    def shape_partition(part):
        return [shape_features(p) for p in part]
    
    def intensity_partition(part):
        return [intensity_features(p) for p in part]
    
    def texture_partition(part):
        return [texture_features(p) for p in part]

    return (
        images
            .map_partitions(shape_partition)
            .map_partitions(intensity_partition)
            .map_partitions(texture_partition)
            
    )