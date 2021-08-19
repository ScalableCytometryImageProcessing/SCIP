from scip.data_loading import multiframe_tiff
from scip.data_masking import mask_apply, mask_creation
from scip.quality_control import intensity_distribution
from scip.data_normalization import quantile_normalization
from scip.main import masked_intensities_partition
import numpy


def test_distributed_partitioned_quantile(data, cluster):
    images, _ = multiframe_tiff.bag_from_directory(
        data, idx=0, channels=[0, 1, 2], partition_size=2)
    bags = mask_creation.create_masks_on_bag(images, noisy_channels=[0])
    bag = bags["otsu"].map_partitions(masked_intensities_partition)

    quantiles = intensity_distribution.get_distributed_partitioned_quantile(
        bag, 0.05, 0.95
    )

    quantiles = quantiles.compute()
    quantiles = numpy.nanmedian(quantiles, axis=-1)
    assert quantiles.shape == (3, 2)
    assert all([q[0] < q[1] for q in quantiles])


def test_quantile_normalization(images_folder, cluster):
    bag, _ = multiframe_tiff.bag_from_directory(
        images_folder, idx=0, channels=[0, 1, 2], partition_size=2)
    bags = mask_creation.create_masks_on_bag(bag, noisy_channels=[0])
    bag = bags["otsu"].map_partitions(masked_intensities_partition)
    bag = quantile_normalization.quantile_normalization(bag, 0.05, 0.95)

    bag = bag.compute()
