from sip.data_loading import multiframe_tiff


def test_correct_amount_of_channels(data, config):
    bag = multiframe_tiff.bag_from_directory(
        data, partition_size=2, channels=config["data_loading"]["channels"])
    images = bag.compute()
    assert len(images[0]["pixels"]) == len(config["data_loading"]["channels"])