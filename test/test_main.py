import pytest
from scip import main
from pathlib import Path
import anndata


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
        partition_size=5,
        with_replacement=with_replacement
    )

    assert runtime is not None
    assert len([f for f in tmp_path.glob("*.h5ad")]) == 10
    assert (tmp_path / "scip.log").exists()
    assert sum(len(anndata.read(f)) for f in tmp_path.glob("*.h5ad")) == expected_n
