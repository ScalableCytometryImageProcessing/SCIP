from typing import List, Mapping, Any


def _bbox_features_meta(channel_names: List[str]) -> Mapping[str, type]:
    d = {
        "bbox_minr": float,
        "bbox_minc": float,
        "bbox_maxr": float,
        "bbox_maxc": float
    }
    d.update({f"regions_{i}": float for i in channel_names})
    return d


def bbox_features(p: Mapping) -> Mapping[str, Any]:
    """Extracts bbox features from image.

    The bbox consist of four values: bbox_minr, bbox_minc, bbox_maxr, bbox_maxc.

    Args:
        p: Contains a sequence of 4 numbers under key bbox.

    Returns:
        Extracted features.
    """

    return list(p["bbox"]) + p["regions"]
