from click.testing import CliRunner
from scip.main import cli
import json
import pandas
import pyarrow.parquet


def test_cli(zarr_path, tmp_path, data):
    runner = CliRunner()
    result = runner.invoke(cli, [
        "--mode", "local",
        "--headless",
        "--n-workers", "4",
        "--n-threads", "1",
        "--n-partitions", "5",
        "--timing", str(tmp_path / "timing.json"),
        str(tmp_path),
        str(data / "scip_zarr.yml"),
        str(zarr_path)
    ])

    assert result.exit_code == 0
    assert (tmp_path / "timing.json").exists()
    with open(tmp_path / "timing.json", "r") as fh:
        timing = json.load(fh)
    assert type(timing["runtime"]) is float
    assert timing["runtime"] > 0

    assert len([f for f in tmp_path.glob("*.parquet")]) == 10
    assert (tmp_path / "scip.log").exists()

    df = pandas.concat(
        [pyarrow.parquet.read_table(f).to_pandas() for f in tmp_path.glob("*.parquet")], axis=0)
    assert len(df) == 10
