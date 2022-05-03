from scip.loading import tiff


def test_bag_from_directory(tiffs_folder):
    bag, futures, meta, _ = tiff.bag_from_directory(
        path = tiffs_folder,
        channels = [1, 2],
        partition_size=1,
        gpu_accelerated=False,
        regex="^.+\/(?P<id>.+)_w(?P<channel>[0-9]).+$",
        output=None,
        segment_method="cellpose",
        segment_kw=dict(
            segmentation_channel_indices=[0,1], 
            dapi_channel_index=None,
            cellpose_segmentation_index=1,
            export=False
        )
    )

    bag = bag.compute()

    assert len(bag) > 0
    assert all("pixels" in b for b in bag)
    assert all(len(b["pixels"].shape) == 3 for b in bag)
    assert all(b["pixels"].shape[0] == 2 for b in bag)

    