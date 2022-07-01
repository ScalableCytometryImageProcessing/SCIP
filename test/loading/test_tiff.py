from scip.loading import tiff
import pytest
import os


@pytest.mark.skipif(
    "GITHUB_ACTIONS" in os.environ,
    reason="Bug in CellPose package related to CPNet on CPU"
)
def test_bag_from_directory(tiffs_folder):
    bag = tiff.bag_from_directory(
        path=tiffs_folder,
        channels=[1, 2],
        partition_size=1,
        gpu_accelerated=False,
        regex="^.+/test(?P<id>.+)_(?P<channel>[0-9]).+$",
        output=None,
        segment_method="cellpose",
        segment_kw=dict(
            cell_diameter=30,
            dapi_channel_index=None,
            segmentation_channel_index=1,
            export=False
        )
    )

    bag = bag.compute()

    assert len(bag) > 0
    assert all("pixels" in b for b in bag)
    assert all(len(b["pixels"].shape) == 3 for b in bag)
    assert all(b["pixels"].shape[0] == 2 for b in bag)
