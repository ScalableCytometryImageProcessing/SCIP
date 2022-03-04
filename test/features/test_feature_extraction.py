from scip.features import feature_extraction, shape, intensity
from dask.dataframe import DataFrame


def test_extract_features(fake_images_bag, image_nchannels):
    features = feature_extraction.extract_features(
        images=fake_images_bag,
        channel_names=[f"c_{i}" for i in range(image_nchannels)],
        types=["bbox", "intensity", "shape", "texture"]
    )

    assert type(features) is DataFrame

    computed = features.compute()

    assert len(computed) == 10
    assert not computed.isna().any(axis=None)


def test_feature_values(fake_images_bag, image_nchannels):
    features = feature_extraction.extract_features(
        images=fake_images_bag,
        channel_names=[f"c_{i}" for i in range(image_nchannels)],
        types=["intensity"]
    )

    computed = features.compute()

    computed.iloc[0]


def test_shape_features(images_bag, image_nchannels):
    bag = images_bag.filter(lambda a: "pixels" in a)
    bag = bag.map(shape.shape_features)
    bag = bag.compute()

    assert bag[0].shape == (len(shape.prop_names) * image_nchannels,)


def test_intensity_features(images_bag, image_nchannels):
    bag = images_bag.filter(lambda a: "pixels" in a)
    bag = bag.map(intensity.intensity_features)
    bag = bag.compute()

    assert bag[0].shape == (len(intensity.props) * image_nchannels * 2,)
