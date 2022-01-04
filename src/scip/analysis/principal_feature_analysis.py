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

import pandas as pd
import dask
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler


class PFA(object):
    def __init__(self, diff_n_features=2, q=None, explained_var=0.95):
        self.q = q
        self.diff_n_features = diff_n_features
        self.explained_var = explained_var

    def fit(self, X):
        sc = StandardScaler()
        X = sc.fit_transform(X)

        pca = PCA().fit(X)
        if not self.q:
            explained_variance = pca.explained_variance_ratio_
            cumulative_expl_var = [sum(explained_variance[:i + 1])
                                   for i in range(len(explained_variance))]
            for i, j in enumerate(cumulative_expl_var):
                if j >= self.explained_var:
                    q = i
                    break

        A_q = pca.components_.T[:, :q]

        clusternumber = min([q + self.diff_n_features, X.shape[1]])

        kmeans = KMeans(n_clusters=clusternumber).fit(A_q)
        clusters = kmeans.predict(A_q)
        cluster_centers = kmeans.cluster_centers_

        dists = defaultdict(list)
        for i, c in enumerate(clusters):
            dist = euclidean_distances([A_q[i, :]], [cluster_centers[c, :]])[0][0]
            dists[c].append((i, dist))

        self.indices_ = [sorted(f, key=lambda x: x[1])[0][0] for f in dists.values()]
        self.features_ = X[:, self.indices_]

    def fit_transform(self, X):
        sc = StandardScaler()
        X = sc.fit_transform(X)

        pca = PCA().fit(X)

        if not self.q:
            explained_variance = pca.explained_variance_ratio_
            cumulative_expl_var = [sum(explained_variance[:i + 1])
                                   for i in range(len(explained_variance))]
            for i, j in enumerate(cumulative_expl_var):
                if j >= self.explained_var:
                    q = i
                    break

        A_q = pca.components_.T[:, :q]

        clusternumber = min([q + self.diff_n_features, X.shape[1]])

        kmeans = KMeans(n_clusters=clusternumber).fit(A_q)
        clusters = kmeans.predict(A_q)
        cluster_centers = kmeans.cluster_centers_

        dists = defaultdict(list)
        for i, c in enumerate(clusters):
            dist = euclidean_distances([A_q[i, :]], [cluster_centers[c, :]])[0][0]
            dists[c].append((i, dist))

        self.indices_ = [sorted(f, key=lambda x: x[1])[0][0] for f in dists.values()]
        self.features_ = X[:, self.indices_]

        return X[:, self.indices_]

    def transform(self, X):
        return X[:, self.indices_]


@dask.delayed
def apply_pfa(features):
    pfa = PFA(diff_n_features=1, explained_var=0.90)
    pfa_result = pfa.fit_transform(features)

    featurekeys = [features.columns.tolist()[i] for i in pfa.indices_]
    pfa_results_df = pd.DataFrame(pfa_result, columns=featurekeys)
    return pfa_results_df
