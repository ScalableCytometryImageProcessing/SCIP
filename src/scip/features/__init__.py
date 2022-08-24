# Copyright (C) 2022 Maxim Lippeveld
#
# This file is part of SCIP.
#
# SCIP is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SCIP is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with SCIP.  If not, see <http://www.gnu.org/licenses/>.

from typing import Iterable, Mapping, Optional

import numpy
import dask
import dask.bag
import dask.dataframe

from .shape import shape_features, _shape_features_meta
from .intensity import intensity_features, _intensity_features_meta
from .texture import texture_features, _texture_features_meta
from .bbox import bbox_features, _bbox_features_meta


def compute_features(images, channel_names, types, loader_meta, prefix):

    def rename(c):
        if any(c.startswith(a) for a in list(loader_meta.keys())):
            return f"meta_{c}"
        elif any(c.startswith(a) for a in ["bbox", "regions"]):
            if prefix is not None:
                return f"meta_{prefix}_{c}"
            else:
                return f"meta_{c}"
        else:
            if prefix is not None:
                return f"feat_{prefix}_{c}"
            else:
                return f"feat_{c}"

    features = extract_features(
        images=images,
        channel_names=channel_names,
        types=types,
        loader_meta=loader_meta
    )
    features = features.rename(columns=rename)
    return features


def features_partition(
    part: Iterable[dict],
    *,
    loader_meta_keys: Iterable[str],
    types: Iterable[str],
    lengths: Mapping[str, int]
):

    out = numpy.full(shape=(len(part), lengths["full"]), fill_value=None, dtype=object)
    for i, p in enumerate(part):
        for j, k in enumerate(loader_meta_keys):
            out[i, j] = p[k]
        c = len(loader_meta_keys)

        if "pixels" in p:
            if "bbox" in types:
                out[i, c:c + lengths["bbox"]] = bbox_features(p)
                c += lengths["bbox"]
            if "shape" in types:
                out[i, c:c + lengths["shape"]] = shape_features(
                    mask=p["mask"],
                    combined_mask=p["combined_mask"]
                )
                c += lengths["shape"]
            if "intensity" in types:
                out[i, c:c + lengths["intensity"]] = intensity_features(
                    pixels=p["pixels"],
                    mask=p["mask"],
                    combined_mask=p["combined_mask"],
                    background=p["background"],
                    combined_background=p["combined_background"]
                )
                c += lengths["intensity"]
            if "texture" in types:
                out[i, c:c + lengths["texture"]] = texture_features(p)

    return out


def extract_features(  # noqa: C901
    *,
    images: dask.bag.Bag,
    channel_names: list,
    types: list,
    loader_meta: Optional[Mapping[str, type]] = {}
) -> dask.dataframe.DataFrame:
    """Extracts requested features from pixel values in images.

    Keyword Args:
        images: bag of mappings containing image data. Check each feature
          extraction method (:func:`bbox_features`,
          :func:`scip.features.intensity.intensity_features`,
          :func:`scip.features.shape.shape_features` and
          :func:`scip.features.texture.texture_features`)
          to see what keys must be present in each mapping.
        channel_names: names of channels in the image.
        types: feature types to be extracted from the image.
        maximum_pixel_value: theoretical maximal value in the image.
        loader_meta: data type mapping of meta keys extracted
          by the loader. Defaults to {}.

    Returns:
        dataframe containing all extracted features (columns) for all
          images (rows) in the input bag.
    """

    metas = {}
    if "bbox" in types:
        metas["bbox"] = _bbox_features_meta(channel_names)
    if "shape" in types:
        metas["shape"] = _shape_features_meta(channel_names)
    if "intensity" in types:
        metas["intensity"] = _intensity_features_meta(channel_names)
    if "texture" in types:
        metas["texture"] = _texture_features_meta(channel_names)

    full_meta = loader_meta.copy()
    lengths = {}
    for k, v in metas.items():
        full_meta.update(v)
        lengths[k] = len(v)
    lengths["full"] = len(full_meta)

    images = images.map_partitions(
        features_partition,
        loader_meta_keys=list(loader_meta.keys()),
        types=types,
        lengths=lengths
    )

    images_df = images.to_dataframe(meta=full_meta, optimize_graph=False)

    return images_df
