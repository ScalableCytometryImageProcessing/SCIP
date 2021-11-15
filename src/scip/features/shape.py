import numpy
from skimage.measure import label, regionprops_table


def shape_features_meta(nchannels):
    props = [
        "area",
        "convex_area",
        "eccentricity",
        "equivalent_diameter",
        "euler_number",
        "feret_diameter_max",
        "filled_area",
        "inertia_tensor",
        "inertia_tensor_eigvals",
        "major_axis_length",
        "minor_axis_length",
        "moments_hu",
        "orientation",
        "perimeter",
        "perimeter_crofton",
        "solidity"
    ]
    out = {}
    for i in range(nchannels):
        out.update({f"{p}_{i}": float for p in props})
    return out


def shape_features(sample):
    """
        compute regionpropse
    Args:
        sample (dict): dictionary containing image data

    Returns:
        dict: dictionary including new features

    """

    img = sample['mask']

    def channel_features(i):
        label_img = label(img[i])
        props = regionprops_table(
            label_image=label_img,
            properties=(
                "area",
                "convex_area",
                "eccentricity",
                "equivalent_diameter",
                "euler_number",
                "feret_diameter_max",
                "filled_area",
                "inertia_tensor",
                "inertia_tensor_eigvals",
                "major_axis_length",
                "minor_axis_length",
                "moments_hu",
                "orientation",
                "perimeter",
                "perimeter_crofton",
                "solidity"
            )
        )
        return props

    features_dict = {}
    for i in range(len(img)):
        if numpy.any(img[i]):
            props = channel_features(i)
            for k, v in props.items():
                features_dict[f"{k}_{i}"] = numpy.mean(v)
        else:
            # setting proper default values if possible when the mask is empty
            features_dict.update({
                f"area_{i}": 0,
                f"convex_area_{i}": 0,
                f"eccentricity_{i}": None,
                f"equivalent_diameter_{i}": 0,
                f"euler_number_{i}": None,
                f"feret_diameter_max_{i}": 0,
                f"filled_area_{i}": 0,
                f"inertia_tensor_{i}": None,
                f"inertia_tensor_eigvals_{i}": None,
                f"major_axis_length_{i}": 0,
                f"minor_axis_length_{i}": 0,
                f"moments_hu_{i}": None,
                f"orientation_{i}": None,
                f"perimeter_{i}": 0,
                f"perimeter_crofton_{i}": 0,
                f"solidity_{i}": None
            })

    return features_dict
