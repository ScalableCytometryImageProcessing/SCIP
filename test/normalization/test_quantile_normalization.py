from scip.loading import multiframe_tiff
from scip.masking import threshold
from scip.normalization import quantile_normalization
import dask.bag
import pickle
import numpy
import pytest


@pytest.mark.skip(reason="currently not working")
def test_distributed_partitioned_quantile(data, cluster):
    with open(str(data / "masked.pickle"), "rb") as fh:
        bag = dask.bag.from_sequence(pickle.load(fh), npartitions=2)
    quantiles = quantile_normalization.get_distributed_partitioned_quantile(
        bag, 0.05, 0.95
    )

    observed_quantiles = quantiles.compute()
    expected_quantiles = numpy.load(str(data / "quantiles.npy"))

    errors = numpy.load(str(data / "quantile_errors.npy"))
    errors = numpy.tile(errors, (2, 1)).T

    assert numpy.all(numpy.abs(observed_quantiles - expected_quantiles) < errors)


@pytest.mark.skip(reason="currently not working")
def test_quantile_normalization(images_folder, cluster):
    bag, _ = multiframe_tiff.bag_from_directory(
        images_folder, idx=0, channels=[0, 1, 2], partition_size=2)
    bag = threshold.create_masks_on_bag(bag, main=True, main_channel=0)
    bag, quantiles = quantile_normalization.quantile_normalization(bag, 0.05, 0.95, 3)

    bag = bag.compute()
    quantiles.compute()


def test_minmax_normalization(images_folder, cluster):
    bag, _ = multiframe_tiff.bag_from_directory(
        images_folder, idx=0, channels=[0, 1, 2], partition_size=2)
    bag = threshold.create_masks_on_bag(bag, main=True, main_channel=0)
    bag, quantiles = quantile_normalization.quantile_normalization(bag, 0, 1, 3)

    bag = bag.compute()
    quantiles = quantiles.compute()
