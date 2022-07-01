import dask.dataframe
from pathlib import Path
from concurrent.futures import Future


def export(
    *,
    df: dask.dataframe.DataFrame,
    output: Path,
    filename: str
) -> Future:
    """Exports dataframe to one AnnData .parquet-file per partition.

    Keyword args:
        df: Dataframe to be exported.
        output: Path to directory where objects should be stored.
        filename: Filename to give to partition
          objects (will be named with format string {filename}.{partition}.h5ad).

    Returns:
        Future that represents the export task.
    """

    return df.to_parquet(
        str(output),
        name_function=lambda x: f"{filename}.{x}.parquet",
        write_metadata_file=False,
        engine="pyarrow"
    )
