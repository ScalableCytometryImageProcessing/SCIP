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
    return features_filtered


@dask.delayed
def feature_stats_to_html(var, mean):
    df = pd.concat([mean, var], axis=1)
    df.columns = ['means', 'var']
    html = df.to_html()
    text_file = open("Quality_report_features2.html", "w")
    text_file.write('<header><h1>UMAP Feature reduction </h1></header>')
    text_file.write(html)
    text_file.close()
    return True


@dask.delayed
def plot_UMAP_to_html(feature_df, table_written):
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

    if table_written:
        html = '<img src=\'data:image/png;base64,{}\'>'.format(encoded)
        text_file = open("Quality_report_features2.html", "a")
        text_file.write('<header><h1>UMAP Feature reduction </h1></header>')
        text_file.write(html)
        text_file.close()
        return True
    return False


def check_report(df, plotted, meta):

    def check_report(part, plotted):
        if plotted:
            return part

    return df.map_partitions(check_report, plotted, meta=meta)


def get_feature_statistics(feature_df):

    var = feature_df.var(axis=0, skipna=True)
    mean = feature_df.mean(axis=0, skipna=True)

    feature_df = filter_features(feature_df, var)
    table_written = feature_stats_to_html(var, mean)
    plotted = plot_UMAP_to_html(feature_df, table_written)
    return plotted, feature_df
