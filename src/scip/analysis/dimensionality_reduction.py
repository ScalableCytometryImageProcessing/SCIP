def umap(df):
    reducer = umap.UMAP()
    values = StandardScaler().fit_transform(df.values)
    embedding = reducer.fit_transform(values)
    return 