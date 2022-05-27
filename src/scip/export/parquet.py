import dask.dataframe
from pathlib import Path
from concurrent.futures import Future


def export(
    *,
    df: dask.dataframe.DataFrame,
    output: Path,
    filename: str
) -> Future:

    return df.to_parquet(
        str(output),
        name_function=lambda x: f"{filename}.{x}.parquet",
        write_metadata_file=False,
        engine="pyarrow"
    )
