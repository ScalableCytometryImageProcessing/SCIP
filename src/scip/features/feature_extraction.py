import dask
import dask.bag
import dask.dataframe

from .shape import shape_features, shape_features_meta
from .intensity import intensity_features, intensity_features_meta
from .texture import texture_features, texture_features_meta


def bbox_features_meta(nchannels):
    d = {
        "bbox_minr": float,
        "bbox_minc": float,
        "bbox_maxr": float,
        "bbox_maxc": float
    }
    d.update({f"regions_{i}": int for i in range(nchannels)})
    return d


def bbox_features(p):
    d = {
        "bbox_minr": p["bbox"][0],
        "bbox_minc": p["bbox"][1],
        "bbox_maxr": p["bbox"][2],
        "bbox_maxc": p["bbox"][3],
    }
    d.update({f"regions_{i}": c for i, c in enumerate(p["regions"])})
    return d


def extract_features(
    *,
    images: dask.bag.Bag,
    nchannels: int,
    types: list,
    maximum_pixel_value: int
):  # noqa: C901
    """
    Extract features from pixel data

    Args:
        images (dask.bag): bag containing dictionaries of image data

    Returns:
        dask.bag: bag containing dictionaries of image features
    """

    def features_partition(part):
        data = []
        for p in part:
            type_dicts = []
            if "bbox" in types:
                type_dicts.append(bbox_features(p))
            if "shape" in types:
                type_dicts.append(shape_features(p))
            if "intensity" in types:
                type_dicts.append(intensity_features(p))
            if "texture" in types:
                type_dicts.append(texture_features(p, maximum_pixel_value))

            out = {"idx": p["idx"]}
            for type_dict in type_dicts:
                out.update(type_dict)
            data.append(out)
        return data

    images = images.map_partitions(features_partition)

    meta = {"idx": str}
    if "bbox" in types:
        meta.update(bbox_features_meta(nchannels))
    if "shape" in types:
        meta.update(shape_features_meta(nchannels))
    if "intensity" in types:
        meta.update(intensity_features_meta(nchannels))
    if "texture" in types:
        meta.update(texture_features_meta(nchannels))
    images_df = images.to_dataframe(meta=meta)

    return images_df
