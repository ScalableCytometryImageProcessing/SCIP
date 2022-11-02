import numpy
from scipy.stats import normaltest


def filter(
    x: numpy.ndarray
) -> bool:
    """Performs normality test on pixel values at .05 significance level.
    Returns True if pixels are not normally distributed."""
    return normaltest(x.ravel()).pvalue < 0.05
