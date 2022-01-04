from typing import Mapping, List, Any

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
    "moments_central-0-0",
    "moments_central-0-1",
    "moments_central-0-2",
    "moments_central-0-3",
    "moments_central-1-0",
    "moments_central-1-1",
    "moments_central-1-2",
    "moments_central-1-3",
    "moments_central-2-0",
    "moments_central-2-1",
    "moments_central-2-2",
    "moments_central-2-3",
    "moments_central-3-0",
    "moments_central-3-1",
    "moments_central-3-2",
    "moments_central-3-3",
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


def _shape_features_meta(channel_names: List[str]) -> Mapping[str, type]:
    out = {}
    for name in channel_names + ["combined"]:
        out.update({f"{p}_{name}": float for p in prop_names})
    return out


def shape_features(sample: Mapping[str, Any], channel_names: List[str]) -> Mapping[str, Any]:
    """Extracts shape features from image.

    The shape features are extracted using :func:regionprops from scikit-image. These include
    features like eccentricity, convex area or equivalent diameter.

    Args:
        sample (Mapping[str, Any]): mapping with mask and combined mask keys.
        channel_names (List[str]): names of channels in the image.

    Returns:
        Mapping[str, Any]: extracted shape features.
    """

    img = sample['mask']

    def _row(mask):
        label_img = label(mask)
        props = regionprops_table(label_image=label_img, properties=prop_ids)
        return props

    features_dict = {}
    props = _row(sample["combined_mask"])
    for k, v in props.items():
        features_dict[f"{k}_combined"] = numpy.mean(v)

    for i, name in enumerate(channel_names):
        if numpy.any(img[i]):
            props = _row(img[i])
            for k, v in props.items():
                features_dict[f"{k}_{name}"] = numpy.mean(v)
        else:
            # setting proper default values if possible when the mask is empty
            features_dict.update({
                f"area_{name}": 0,
                f"convex_area_{name}": 0,
                f"equivalent_diameter_{name}": 0,
                f"feret_diameter_max_{name}": 0,
                f"filled_area_{name}": 0,
                f"major_axis_length_{name}": 0,
                f"minor_axis_length_{name}": 0,
                f"perimeter_{name}": 0,
                f"perimeter_crofton_{name}": 0,
                f"solidity_{name}": 0,
                f"extent_{name}": 0,
                f"eccentricity_{name}": None,
                f"euler_number_{name}": None,
                f"orientation_{name}": None,
                f"inertia_tensor-0-0_{name}": None,
                f"inertia_tensor-0-1_{name}": None,
                f"inertia_tensor-1-0_{name}": None,
                f"inertia_tensor-1-1_{name}": None,
                f"inertia_tensor_eigvals-0_{name}": None,
                f"inertia_tensor_eigvals-1_{name}": None,
                f"moments-0-0_{name}": None,
                f"moments-0-1_{name}": None,
                f"moments-0-2_{name}": None,
                f"moments-0-3_{name}": None,
                f"moments-1-0_{name}": None,
                f"moments-1-1_{name}": None,
                f"moments-1-2_{name}": None,
                f"moments-1-3_{name}": None,
                f"moments-2-0_{name}": None,
                f"moments-2-1_{name}": None,
                f"moments-2-2_{name}": None,
                f"moments-2-3_{name}": None,
                f"moments-3-0_{name}": None,
                f"moments-3-1_{name}": None,
                f"moments-3-2_{name}": None,
                f"moments-3-3_{name}": None,
                f"moments_central-0-0_{name}": None,
                f"moments_central-0-1_{name}": None,
                f"moments_central-0-2_{name}": None,
                f"moments_central-0-3_{name}": None,
                f"moments_central-1-0_{name}": None,
                f"moments_central-1-1_{name}": None,
                f"moments_central-1-2_{name}": None,
                f"moments_central-1-3_{name}": None,
                f"moments_central-2-0_{name}": None,
                f"moments_central-2-1_{name}": None,
                f"moments_central-2-2_{name}": None,
                f"moments_central-2-3_{name}": None,
                f"moments_central-3-0_{name}": None,
                f"moments_central-3-1_{name}": None,
                f"moments_central-3-2_{name}": None,
                f"moments_central-3-3_{name}": None,
                f"moments_hu-0_{name}": None,
                f"moments_hu-1_{name}": None,
                f"moments_hu-2_{name}": None,
                f"moments_hu-3_{name}": None,
                f"moments_hu-4_{name}": None,
                f"moments_hu-5_{name}": None,
                f"moments_hu-6_{name}": None
            })

    return features_dict
