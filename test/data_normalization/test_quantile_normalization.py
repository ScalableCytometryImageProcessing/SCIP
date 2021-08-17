from scip.data_loading import multiframe_tiff
from scip.data_masking import mask_apply, mask_creation
from scip.quality_control import intensity_distribution
from scip.data_normalization import quantile_normalization


def test_distributed_partitioned_quantile(data, cluster):
    bag = multiframe_tiff.bag_from_directory(data, channels=[0, 1, 2], partition_size=2)
    bag = mask_creation.create_masks_on_bag(bag, noisy_channels=[0])
    bag = mask_apply.create_masked_images_on_bag(bag)

    quantiles, masked_quantiles = intensity_distribution.get_distributed_partitioned_quantile(
        bag, 0.05, 0.95
    )

    quantiles = quantiles.compute()
    masked_quantiles = masked_quantiles.compute()

    assert quantiles.shape == masked_quantiles.shape
    assert quantiles.shape == (3, 2)
    assert all([q[0] < q[1] for q in quantiles])
    assert all([q[0] < q[1] for q in masked_quantiles])


def test_quantile_normalization(data, cluster):
    bag = multiframe_tiff.bag_from_directory(data, channels=[0, 1, 2], partition_size=2)
    bag = mask_creation.create_masks_on_bag(bag, noisy_channels=[0])
    bag = mask_apply.create_masked_images_on_bag(bag)

    bag = quantile_normalization.quantile_normalization(bag, 0.05, 0.95)

    bag.compute()
