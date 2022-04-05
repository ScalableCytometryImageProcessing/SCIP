from scip.features import feature_extraction
from dask.dataframe import DataFrame
import pytest


@pytest.mark.parametrize("fake_images_bag", [True], indirect=True)
def test_extract_features(fake_images_bag, fake_image_nchannels):
    features = feature_extraction.extract_features(
        images=fake_images_bag,
        channel_names=[f"c_{i}" for i in range(fake_image_nchannels)],
        types=["bbox", "intensity", "shape", "texture"]
    )

    assert type(features) is DataFrame

    computed = features.compute()

    assert len(computed) == 10
    assert not computed.isna().any(axis=None)
