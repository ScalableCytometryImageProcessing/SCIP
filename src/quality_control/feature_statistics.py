import numpy as np
import dask


def get_shape_features(sample):
    features = sample.get('shape_features')

    combined = np.array([])
    for channel in features:
        combined = np.hstack((combined, list(channel.values())))

    return combined


def stack_features(L1, L2):
    return np.vstack((L1, L2))


@dask.delayed
def mean_calculation(stacked):
    return np.nanmean(stacked, axis=0)


@dask.delayed
def variance_calculation(stacked):
    return np.nanvar(stacked, axis=0)


@dask.delayed
def median_calculation(stacked):
    return np.nanmedian(stacked, axis=0)


def shape_partition(part):

    stacked = np.array([])
    for p in part:
        stacked = np.vstack((stacked, get_shape_features(p)))

    # Calculate stats on stacked samples
    mean = np.nanmean(stacked, axis=0)
    median = np.nanmedian(stacked, axis=0)
    var = np.nanvar(stacked, axis=0)
    return mean, median, var


def get_feature_statistics(feature_df):

    mean = feature_df.mean(axis=0, skipna=True)
    var = feature_df.var(axis=0, skipna=True)
    columns = feature_df.columns

    return mean, var, columns
