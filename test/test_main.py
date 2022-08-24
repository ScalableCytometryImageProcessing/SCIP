import pytest
from scip import main
from pathlib import Path
import pyarrow.parquet
import pandas
import numpy


@pytest.mark.parametrize(
    'mode,limit,expected_n,with_replacement',
    [
        pytest.param('local', 2, 2, False, marks=pytest.mark.mpi_skip),
        pytest.param('local', 8, 8, False, marks=pytest.mark.mpi_skip),
        pytest.param('local', -1, 10, False, marks=pytest.mark.mpi_skip),
        pytest.param('local', 20, 20, True, marks=pytest.mark.mpi_skip),
        pytest.param('mpi', -1, 10, False, marks=[
            pytest.mark.mpi, pytest.mark.skip(reason="Issues with MPI")])
    ])
def test_main(mode, limit, expected_n, with_replacement, zarr_path, tmp_path, data):
    runtime = main.main(
        mode=mode,
        n_workers=4,
        n_threads=1,
        headless=True,
        output=Path(tmp_path),
        paths=[str(zarr_path)],
        config=data / "scip_zarr.yml",
        limit=limit,
        n_partitions=2,
        with_replacement=with_replacement
    )

    assert runtime is not None
    assert len([f for f in tmp_path.glob("*.parquet")]) == 10
    assert (tmp_path / "scip.log").exists()

    df = pandas.concat(
        [pyarrow.parquet.read_table(f).to_pandas() for f in tmp_path.glob("*.parquet")], axis=0)
    assert len(df) == expected_n

    cols = df.columns
    assert sum("circle-1" in a for a in cols) > 0
    assert sum("circle-2" in a for a in cols) > 0
    assert sum("spot" in a for a in cols) > 0
    assert sum("circle-1" in a for a in cols) == sum("circle-2" in a for a in cols)
    assert sum("circle-1" in a for a in cols) == sum("spot" in a for a in cols)

    assert numpy.all(df.filter(regex="circle-1").values == df.filter(regex="circle-2").values)


def test_main_with_correction(tiffs_folder, tmp_path, data):
    runtime = main.main(
        mode="local",
        n_workers=4,
        n_threads=1,
        headless=True,
        output=Path(tmp_path),
        paths=[str(tiffs_folder)],
        config=data / "scip_tiff_seg.yml",
        n_partitions=2
    )

    assert runtime is not None
    assert len([f for f in tmp_path.glob("*.parquet")]) == 10
    assert (tmp_path / "scip.log").exists()

    df = pandas.concat(
        [pyarrow.parquet.read_table(f).to_pandas() for f in tmp_path.glob("*.parquet")], axis=0)
    assert len(df) > 0

    assert (tmp_path / "correction_images.pickle").exists()
