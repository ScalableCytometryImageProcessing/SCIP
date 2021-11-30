import numpy
from skimage.measure import label, regionprops_table

prop_names = [
    "area",
    "convex_area",
    "eccentricity",
    "equivalent_diameter",
    "euler_number",
    "feret_diameter_max",
    "filled_area",
    "major_axis_length",
    "minor_axis_length",
    "orientation",
    "perimeter",
    "perimeter_crofton",
    "solidity",
    "extent",
    "inertia_tensor-0-0",
    "inertia_tensor-0-1",
    "inertia_tensor-1-0",
    "inertia_tensor-1-1",
    "inertia_tensor_eigvals-0",
    "inertia_tensor_eigvals-1",
    "moments-0-0",
    "moments-0-1",
    "moments-0-2",
    "moments-0-3",
    "moments-1-0",
    "moments-1-1",
    "moments-1-2",
    "moments-1-3",
    "moments-2-0",
    "moments-2-1",
    "moments-2-2",
    "moments-2-3",
    "moments-3-0",
    "moments-3-1",
    "moments-3-2",
    "moments-3-3",
    "moments-central-0-0",
    "moments-central-0-1",
    "moments-central-0-2",
    "moments-central-0-3",
    "moments-central-1-0",
    "moments-central-1-1",
    "moments-central-1-2",
    "moments-central-1-3",
    "moments-central-2-0",
    "moments-central-2-1",
    "moments-central-2-2",
    "moments-central-2-3",
    "moments-central-3-0",
    "moments-central-3-1",
    "moments-central-3-2",
    "moments-central-3-3",
    "moments_hu-0",
    "moments_hu-1",
    "moments_hu-2",
    "moments_hu-3",
    "moments_hu-4",
    "moments_hu-5",
    "moments_hu-6"
]
prop_ids = [
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
    "moments",
    "moments_central",
    "moments_hu",
    "orientation",
    "perimeter",
    "perimeter_crofton",
    "solidity",
    "extent"
]

def shape_features_meta(nchannels):
    out = {}
    for i in range(nchannels):
        out.update({f"{p}_{i}": float for p in prop_names})
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
        props = regionprops_table(label_image=label_img, properties=prop_ids)
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
                f"major_axis_length_{i}": 0,
                f"minor_axis_length_{i}": 0,
                f"orientation_{i}": None,
                f"perimeter_{i}": 0,
                f"perimeter_crofton_{i}": 0,
                f"solidity_{i}": 0,
                f"extent_{i}": 0,
                f"inertia_tensor-0-0": None,
                f"inertia_tensor-0-1": None,
                f"inertia_tensor-1-0": None,
                f"inertia_tensor-1-1": None,
                f"inertia_tensor_eigvals-0": None,
                f"inertia_tensor_eigvals-1": None,
                f"moments-0-0": None,
                f"moments-0-1": None,
                f"moments-0-2": None,
                f"moments-0-3": None,
                f"moments-1-0": None,
                f"moments-1-1": None,
                f"moments-1-2": None,
                f"moments-1-3": None,
                f"moments-2-0": None,
                f"moments-2-1": None,
                f"moments-2-2": None,
                f"moments-2-3": None,
                f"moments-3-0": None,
                f"moments-3-1": None,
                f"moments-3-2": None,
                f"moments-3-3": None,
                f"moments-central-0-0": None,
                f"moments-central-0-1": None,
                f"moments-central-0-2": None,
                f"moments-central-0-3": None,
                f"moments-central-1-0": None,
                f"moments-central-1-1": None,
                f"moments-central-1-2": None,
                f"moments-central-1-3": None,
                f"moments-central-2-0": None,
                f"moments-central-2-1": None,
                f"moments-central-2-2": None,
                f"moments-central-2-3": None,
                f"moments-central-3-0": None,
                f"moments-central-3-1": None,
                f"moments-central-3-2": None,
                f"moments-central-3-3": None,
                f"moments_hu-0": None,
                f"moments_hu-1": None,
                f"moments_hu-2": None,
                f"moments_hu-3": None,
                f"moments_hu-4": None,
                f"moments_hu-5": None,
                f"moments_hu-6": None
            })

    return features_dict
