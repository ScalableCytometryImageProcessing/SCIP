import dask
from dask_ml.decomposition import PCA
import pandas as pd
import dask.dataframe as dd

import numpy as np
    

def dimensionality_reduction(features):
    filterd_features = features.drop(columns=['path'])
    pca = PCA(n_components=3)
    reduced_features = pca.fit_transform(filterd_features.to_dask_array(lengths=True))
    return reduced_features.to_dask_dataframe()


def distance(center, point):
    """ Calculate distance between point and center """
    return np.sqrt(np.sum((np.array(point) - center) ** 2))


def get_membership(sample, centers, m):
    centers_np = centers.to_numpy()
    distances = list(map(lambda cluster: pow(distance(cluster, sample), (-2.0 / (m - 1))), centers_np))
    distance_sum = sum(distances)
    
    membership = [j / distance_sum for j in centers_np]
    dx = [np.multiply(np.power(j, m), sample) for j in membership]
    x = [np.power(j, m) for j in membership]
    return (dx, x)

        
def fuzzy_c_means(features, amount_of_centers, m, iterations):
    
    def get_initial_centers(samples):
        # When working with small datasets, amount of samples will be capped to the
        # amount of samples kept in a partition
        centers = samples.head(n=5, compute=False)
        
        # Add offset to centers to avoid infinite memberships
        return centers + 1

        
    def get_membership(row, center_points, m):
        distances = (center_points - row).pow(2, axis=1).sum(axis=1).pow(0.5)
        dist_sum = distances.sum()
        non_normalized = ((distances / dist_sum).pow(-2 / (m - 1)))
        return (non_normalized / non_normalized.sum())


    def get_numerator_partition(part, centers, m, amount_of_centers):
        membership =  part.apply(get_membership, center_points=centers, m=m, axis=1)
        fuzzied_membership = membership.pow(m)
        center_list = []
        for column in fuzzied_membership:
            center_list.append(part.mul(fuzzied_membership[column], axis=0).apply(lambda x: x.sum()))
            #  temp = temp / total_sum[column]
        df =  pd.DataFrame(center_list)
        df.index.name = "centers"
        return df

    def get_memberships_partioned(part, centers, m, amount_of_centers):
        membership =  part.apply(get_membership, center_points=centers, m=m, axis=1)
        return membership



    pca_df = dimensionality_reduction(features)
    centers = get_initial_centers(pca_df)
    for j in range(iterations):
        numerator_collection = pca_df.map_partitions(get_numerator_partition, centers=centers, m=m, amount_of_centers=amount_of_centers)
        summed_numerators = numerator_collection.groupby('centers').sum()
        memberships = pca_df.map_partitions(get_memberships_partioned, centers=centers, m=m, amount_of_centers=amount_of_centers, meta={0:float, 1:float, 2:float, 3:float, 4:float})
        summed_denominator = memberships.pow(m).reduction(lambda x: x.sum())
        
        centers = summed_numerators.div(summed_denominator, axis=0)
        
    return memberships
