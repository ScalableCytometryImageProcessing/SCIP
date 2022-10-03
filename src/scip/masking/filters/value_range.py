import numpy


def filter(
    x: numpy.ndarray,
    threshold: float
) -> bool:
    return (x.max() - x.min()) > threshold
