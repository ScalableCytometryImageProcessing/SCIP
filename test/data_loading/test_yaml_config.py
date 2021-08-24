from scip.loading import multiframe_tiff


def test_correct_amount_of_channels(images_folder, config):
    bag, _ = multiframe_tiff.bag_from_directory(
        images_folder, idx=0, partition_size=2, channels=config["loading"]["channels"])
    images = bag.compute()
    assert len(images[0]["pixels"]) == len(config["loading"]["channels"])
