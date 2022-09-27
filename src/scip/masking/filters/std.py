from typing import List
import numpy


def filter(
    x: numpy.ndarray,
    threshold: float
) -> bool:
    return x.std() > threshold
