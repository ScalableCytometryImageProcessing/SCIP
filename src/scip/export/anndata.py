from typing import Mapping

import dask.dataframe
from dask.highlevelgraph import HighLevelGraph
from pathlib import Path
from concurrent.futures import Future
import pandas
import anndata
import numpy


def export(
    *,
    df: dask.dataframe.DataFrame,
    output: Path,
    filename: str
) -> Future:
    """Exports dataframe to one AnnData .h5ad-file per partition.

    Keyword args:
        df: Dataframe to be exported.
        output: Path to directory where objects should be stored.
        filename: Filename to give to partition
          objects (will be named with format string {filename}.{partition}.h5ad).

    Returns:
        Future that represents the export task.
    """

    def _write_anndata(
        df: pandas.DataFrame,
        partition_info: Mapping[str, int]
    ):
        """Creates and writes AnnData object for one partition"""
        x = partition_info["number"]
        df = df.reset_index(drop=True)
        df.index = df.index.astype(str)
        anndata.AnnData(
            X=df.filter(regex="feat"),
            dtype=numpy.float32,
            obs=df.filter(regex="meta")
        ).write(output / f"{filename}.{x}.h5ad")

    data_write = df.map_partitions(
        _write_anndata,
        meta=df._meta,
        enforce_metadata=False,
        transform_divisions=False,
        align_dataframes=False
    )

    # Add noop to graph
    final_name = "store-" + data_write._name
    dsk = {(final_name, 0): (lambda x: None, data_write.__dask_keys__())}

    # Convert data_write + dsk to computable collection
    graph = HighLevelGraph.from_collections(final_name, dsk, dependencies=(data_write,))
    out = dask.dataframe.core.Scalar(graph, final_name, "")

    return out
