"""Functions for extracting features from images.
"""

from typing import Iterable, Mapping, Any

import dask
import dask.bag
import dask.dataframe

from .shape import shape_features, _shape_features_meta
from .intensity import intensity_features, _intensity_features_meta
from .texture import texture_features, _texture_features_meta


def _bbox_features_meta(channel_names: Iterable[str]) -> Mapping[str, type]:
    d = {
        "bbox_minr": float,
        "bbox_minc": float,
        "bbox_maxr": float,
        "bbox_maxc": float
    }
    d.update({f"regions_{i}": float for i in channel_names})
    return d


def bbox_features(p: Mapping, channel_names: Iterable[str]) -> Mapping[str, Any]:
    """Extracts bbox features from image.

    The bbox consist of four values: bbox_minr, bbox_minc, bbox_maxr, bbox_maxc.

    Args:
        p (Mapping): Contains a sequence of 4 numbers under key bbox.
        channel_names (Iterable[str]): names of channels in the image.

    Returns:
        Mapping[str, Any]: extracted features.
    """

    d = {
        "bbox_minr": p["bbox"][0],
        "bbox_minc": p["bbox"][1],
        "bbox_maxr": p["bbox"][2],
        "bbox_maxc": p["bbox"][3],
    }
    d.update({f"regions_{i}": c for i, c in zip(channel_names, p["regions"])})
    return d


def extract_features(  # noqa: C901
    *,
    images: dask.bag.Bag,
    channel_names: list,
    types: list,
    maximum_pixel_value: int,
    loader_meta: Mapping[str, type] = {}
) -> dask.dataframe.DataFrame:
    """Extracts requested features from pixel values in images.

    Keyword Args:
        images (dask.bag.Bag): bag of mappings containing image data. Check each feature
          extraction method (:func:`bbox_features`, :func:`scip.features.intensity.intensity_features`,
          :func:`shape_features` and :func:`texture_features`) to see what keys
          must be present in each mapping.
        channel_names (list): names of channels in the image.
        types (list): feature types to be extracted from the image.
        maximum_pixel_value (int): theoretical maximal value in the image.
        loader_meta (Mapping[str, type], optional): data type mapping of meta keys extracted
          by the loader. Defaults to {}.

    Returns:
        dask.dataframe.DataFrame:
          dataframe containing all extracted features (columns) for all
          images (rows) in the input bag.
    """

    def features_partition(part):
        data = []
        for p in part:
            out = {k: p[k] for k in loader_meta.keys()}

            if "pixels" in p:
                if "bbox" in types:
                    out.update(bbox_features(p, channel_names))
                if "shape" in types:
                    out.update(shape_features(p, channel_names))
                if "intensity" in types:
                    out.update(intensity_features(p, channel_names))
                if "texture" in types:
                    out.update(texture_features(p, channel_names, maximum_pixel_value))

            data.append(out)
        return data

    meta = {}
    if "bbox" in types:
        meta.update(_bbox_features_meta(channel_names))
    if "shape" in types:
        meta.update(_shape_features_meta(channel_names))
    if "intensity" in types:
        meta.update(_intensity_features_meta(channel_names))
    if "texture" in types:
        meta.update(_texture_features_meta(channel_names))

    full_meta = {**meta, **loader_meta}

    images = images.map_partitions(features_partition)
    images_df = images.to_dataframe(meta=full_meta, optimize_graph=False)

    return images_df
