from scip.features import feature_extraction
from dask.dataframe import DataFrame

def test_extract_features(images_bag, image_nchannels):
    features = feature_extraction.extract_features(
        images=images_bag,
        channel_names=[f"c_{i}" for i in range(image_nchannels)],
        types=["bbox", "intensity", "shape", "texture"]
    )

    assert type(features) is DataFrame

    computed = features.compute()

    assert len(computed) == 10
    assert not computed.isna().any(axis=None)
