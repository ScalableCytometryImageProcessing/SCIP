from scip.data_loading import multiframe_tiff
from scip.data_masking import mask_creation
from scip.data_normalization import quantile_normalization
from scip.main import masked_intensities_partition
import dask.bag
import pickle
import numpy


def test_distributed_partitioned_quantile(data, cluster):
    with open(str(data / "masked.pickle"), "rb") as fh:
        bag = dask.bag.from_sequence(pickle.load(fh), npartitions=2)
    bag = bag.map_partitions(masked_intensities_partition)
    quantiles = quantile_normalization.get_distributed_partitioned_quantile(
        bag, 0.05, 0.95
    )
    
    observed_quantiles = quantiles.compute()
    expected_quantiles = numpy.load(str(data / "quantiles.npy"))

    errors = numpy.load(str(data / "quantile_errors.npy"))
    errors = numpy.tile(errors, (2,1)).T

    assert numpy.all(numpy.abs(observed_quantiles - expected_quantiles) < errors)


def test_quantile_normalization(images_folder, cluster):
    bag, _ = multiframe_tiff.bag_from_directory(
        images_folder, idx=0, channels=[0, 1, 2], partition_size=2)
    bags = mask_creation.create_masks_on_bag(bag, noisy_channels=[0])
    bag = bags["otsu"].map_partitions(masked_intensities_partition)
    bag = quantile_normalization.quantile_normalization(bag, 0.05, 0.95)

    bag = bag.compute()
