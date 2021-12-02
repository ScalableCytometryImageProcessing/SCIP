from scip.loading import multiframe_tiff
from scip.segmentation import watershed, util
from functools import partial


def test_bounding_box(images_folder, cluster):

    bag, _ = multiframe_tiff.bag_from_directory(
        images_folder, idx=0, channels=[0, 1, 2], partition_size=2)
    bag = watershed.create_masks_on_bag(bag, noisy_channels=[0])
    bag = bag.filter(partial(util.mask_predicate, bbox_channel_index=0))
    bag = bag.map_partitions(util.bounding_box_partition, bbox_channel_index=0)

    bag = bag.compute()

    for el in bag:
        bbox = el["bbox"]

        assert len(bbox) == 4
        assert all(isinstance(x, int) for x in bbox)
        assert bbox[0] < bbox[2]
        assert bbox[1] < bbox[3]
