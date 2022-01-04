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

import numpy as np
import pandas as pd
from scip.reports.util import get_jinja_template


def filter_features(feature_df, var):

    # Find columns to drop
    zero_variance = list(var[var == 0].index)
    nan_columns = list(var[np.isnan(var)].index)

    # Drop columns
    features_filtered = feature_df.drop(columns=zero_variance)
    features_filtered = features_filtered.drop(columns=nan_columns)

    # Drop rows with nan's
    features_filtered = features_filtered.dropna()
    return features_filtered, zero_variance, nan_columns


def report(df, *, template_dir, template, output):

    features, _ = df.filter(regex="^feat_"), df.filter(regex="^meta_")

    def feature_stats_to_html(var, mean, dropped_zero_variance, dropped_nan, output):
        df = pd.concat([mean, var], axis=1)
        df.columns = ['means', 'var']
        stats = df.to_html()

        # Construct two single columns dataframes with dropped features
        zero_variance = pd.DataFrame(dropped_zero_variance, columns=['feature']).to_html()
        nan = pd.DataFrame(dropped_nan, columns=['feature']).to_html()

        # Write HTML
        with open(str(output / "quality_report_features.html"), "w") as fh:
            fh.write(get_jinja_template(template_dir, template).render(
                stats=stats,
                zero_variance=zero_variance,
                nan=nan
            ))

    var = features.var(axis=0, skipna=True)
    mean = features.mean(axis=0, skipna=True)

    filtered_df, dropped_zero_variance, dropped_nan = filter_features(df, var)

    return feature_stats_to_html(
        var, mean, dropped_zero_variance, dropped_nan, output)
