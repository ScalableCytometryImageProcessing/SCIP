import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import umap
import dask


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


@dask.delayed
def feature_stats_to_html(var, mean, dropped_zero_variance, dropped_nan, feature_df, output):
    df = pd.concat([mean, var], axis=1)
    df.columns = ['means', 'var']
    html = df.to_html()

    # Construct two single columns dataframes with dropped features
    zero_variance_html = pd.DataFrame(dropped_zero_variance, columns=['feature']).to_html()
    nan_html = pd.DataFrame(dropped_nan, columns=['feature']).to_html()

    # Write HTML
    with open(str(output / "quality_report_features.html"), "w") as text_file:
        text_file.write('<header><h1>Feature statistics</h1></header>')
        text_file.write(html)
        text_file.write('<header><h1>Dropped columns: NaN</h1></header>')
        text_file.write(nan_html)
        text_file.write('<header><h1>Dropped columns: Zero variance</h1></header>')
        text_file.write(zero_variance_html)

    reducer = umap.UMAP()
    df_floats = feature_df.drop(columns=['path'])
    scaled_cell_data = StandardScaler().fit_transform(df_floats)
    embedding = reducer.fit_transform(scaled_cell_data)

    fig = plt.figure()
    plt.scatter(embedding[:, 0], embedding[:, 1])
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('UMAP projection of dataset', fontsize=24)
    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')

    with open(str(output / "quality_report_features.html"), "a") as text_file:
        html = '<img src=\'data:image/png;base64,{}\'>'.format(encoded)
        text_file.write('<header><h1>UMAP Feature reduction </h1></header>')
        text_file.write(html)
        text_file.close()


def get_feature_statistics(feature_df, output):

    var = feature_df.var(axis=0, skipna=True)
    mean = feature_df.mean(axis=0, skipna=True)

    feature_df, dropped_zero_variance, dropped_nan = filter_features(feature_df, var)
    return feature_stats_to_html(
        var, mean, dropped_zero_variance, dropped_nan, feature_df, output)
