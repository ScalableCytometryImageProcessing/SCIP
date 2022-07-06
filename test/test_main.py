import pytest
from scip import main
from pathlib import Path
import anndata


@pytest.mark.parametrize(
    'mode',
    [
        pytest.param('local', marks=pytest.mark.mpi_skip),
        pytest.param('mpi', marks=[
            pytest.mark.mpi, pytest.mark.skip(reason="Issues with MPI")])
    ])
def test_main(mode, zarr_path, tmp_path, data):
    runtime = main.main(
        mode=mode,
        n_workers=4,
        n_threads=1,
        headless=True,
        output=Path(tmp_path),
        paths=[str(zarr_path)],
        config=data / "scip_zarr.yml"
    )

    assert runtime is not None
    assert len([f for f in tmp_path.glob("*.h5ad")]) == 10
    assert (tmp_path / "scip.log").exists()
    assert sum(len(anndata.read(f)) for f in tmp_path.glob("*.h5ad")) > 0
