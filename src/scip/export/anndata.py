from typing import Mapping

import dask.dataframe
from dask.highlevelgraph import HighLevelGraph
from pathlib import Path
from concurrent.futures import Future
import pandas
import anndata


def export(
    *,
    df: dask.dataframe.DataFrame,
    output: Path,
    filename: str
) -> Future:

    def _write_anndata(
        df: pandas.DataFrame,
        partition_info: Mapping[str, int]
    ):
        x = partition_info["number"]
        df = df.reset_index(drop=True)
        anndata.AnnData(
            X=df.filter(regex="feat"),
            obs=df.filter(regex="meta")
        ).write(output / f"{filename}.{x}.h5ad")

    data_write = df.map_partitions(
        _write_anndata,
        meta=df._meta,
        enforce_metadata=False,
        transform_divisions=False,
        align_dataframes=False
    )

    final_name = "store-" + data_write._name
    dsk = {(final_name, 0): (lambda x: None, data_write.__dask_keys__())}

    # Convert data_write + dsk to computable collection
    graph = HighLevelGraph.from_collections(final_name, dsk, dependencies=(data_write,))
    out = dask.dataframe.core.Scalar(graph, final_name, "")

    return out
