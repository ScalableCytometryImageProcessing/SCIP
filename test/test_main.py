import pytest
from scip import main
from pathlib import Path
import anndata


@pytest.mark.parametrize(
    'mode,limit',
    [
        pytest.param('local',1, marks=pytest.mark.mpi_skip),
        pytest.param('local',-1, marks=pytest.mark.mpi_skip),
        pytest.param('mpi',1, marks=[
            pytest.mark.mpi, pytest.mark.skip(reason="Issues with MPI")])
    ])
def test_main(mode, limit, zarr_path, tmp_path, data):
    runtime = main.main(
        mode=mode,
        n_workers=4,
        n_threads=1,
        headless=True,
        output=Path(tmp_path),
        paths=[str(zarr_path)],
        config=data / "scip_zarr.yml",
        limit=limit
    )

    assert runtime is not None
    assert len([f for f in tmp_path.glob("*.h5ad")]) == 10
    assert (tmp_path / "scip.log").exists()
    assert sum(len(anndata.read(f)) for f in tmp_path.glob("*.h5ad")) > 0
