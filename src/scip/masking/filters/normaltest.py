import numpy
from scipy.stats import normaltest


def filter(
    x: numpy.ndarray
) -> bool:
    return normaltest(x.ravel()).pvalue < 0.05
