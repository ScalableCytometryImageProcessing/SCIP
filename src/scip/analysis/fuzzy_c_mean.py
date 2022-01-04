# Copyright (C) 2022 Maxim Lippeveld
#
# This file is part of SCIP.
#
# SCIP is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SCIP is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with SCIP.  If not, see <http://www.gnu.org/licenses/>.

import umap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
# Dask imports
import dask
from dask_ml.decomposition import PCA
from io import BytesIO
import base64


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
    distances = list(map(lambda cluster: pow(distance(cluster, sample),
                                                     (-2.0 / (m - 1))), centers_np))
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
        membership = part.apply(get_membership, center_points=centers, m=m, axis=1)
        fuzzied_membership = membership.pow(m)
        center_list = []
        for column in fuzzied_membership:
            center_list.append(part.mul(fuzzied_membership[column], axis=0)
                                   .apply(lambda x: x.sum()))
        df = pd.DataFrame(center_list)
        df.index.name = "centers"
        return df

    def get_memberships_partioned(part, centers, m, amount_of_centers):
        membership = part.apply(get_membership, center_points=centers, m=m, axis=1)
        return membership

    @dask.delayed
    def plot_membership(memberships, features):
        reducer = umap.UMAP()
        scaled_cell_data = StandardScaler().fit_transform(features)
        embedding = reducer.fit_transform(scaled_cell_data)

        scatter_df = pd.DataFrame(embedding)
        rows = memberships.shape[1]
        cols = 1
        fuzzy_c_means_fg, axarr = plt.subplots(rows, cols, figsize=(5, 20))

        for count, column in enumerate(memberships):
            axarr[count].scatter(scatter_df[0], scatter_df[1], c=memberships[column],
                                 cmap='Blues', s=1, alpha=0.5)
            axarr[count].title.set_text(f"UMAP for cluster: {count}")

        fuzzy_cmeans_tmp = BytesIO()
        fuzzy_c_means_fg.savefig(fuzzy_cmeans_tmp, format='png')
        encoded = base64.b64encode(fuzzy_cmeans_tmp.getvalue()).decode('utf-8')
        facet_plots = '<img src=\'data:image/png;base64,{}\'>'.format(encoded)

        # Write HTML
        text_file = open("Fuzzy_clustering.html", "w")
        text_file.write('<header><h1>Fuzzy C mean clustering</h1></header>')
        text_file.write(facet_plots)
        text_file.close()

        return True

    pca_df = dimensionality_reduction(features)
    centers = get_initial_centers(pca_df)

    for j in range(iterations):
        numerator_collection = pca_df.map_partitions(get_numerator_partition, centers=centers,
                                                     m=m, amount_of_centers=amount_of_centers)
        summed_numerators = numerator_collection.groupby('centers').sum()
        memberships = pca_df.map_partitions(get_memberships_partioned, centers=centers, m=m,
                                            amount_of_centers=amount_of_centers,
                                            meta={0: float, 1: float, 2: float, 3: float, 4: float})
        summed_denominator = memberships.pow(m).reduction(lambda x: x.sum())
        centers = summed_numerators.div(summed_denominator, axis=0)

    return memberships, plot_membership(memberships, pca_df)
