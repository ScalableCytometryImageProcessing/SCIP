import numpy


def filter(
    x: numpy.ndarray,
    threshold: float
) -> bool:
    """Checks range width of pixel values against threshold.
    Returns True if width is above threshold."""
    return (x.max() - x.min()) > threshold
