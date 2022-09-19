from scip.loading import load_meta, load_pixels, tiff
from scip.segmentation import watershed_dapi


def test_watershed(tiffs_folder):

    meta = load_meta(
        paths=[tiffs_folder],
        loader_module=tiff,
        kwargs=dict(regex="^.+/(?P<plate>.+)/test(?P<id>.+)_(?P<channel>[0-9]).+$")
    )
    images = load_pixels(
        bag=meta,
        channels=[1, 2],
        loader_module=tiff,
        kwargs=dict(regex="^.+/(?P<plate>.+)/test(?P<id>.+)_(?P<channel>[0-9]).+$")
    )

    images = images.map_partitions(
        watershed_dapi.segment_block, cell_diameter=15, dapi_channel_index=0)

    images = images.compute()

    assert len(images) > 0
    assert all("mask" in e for e in images)
    assert all(e["mask"].max() > 0 for e in images)
    assert all("pixels" in e for e in images)
    assert all(e["pixels"].shape == e["mask"].shape for e in images)
