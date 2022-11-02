import numpy


def filter(
    x: numpy.ndarray,
    threshold: float
) -> bool:
    """Checks standard deviation of pixel values against threshold.
    Returns True if standard deviation is above threshold."""
    return x.std() > threshold
