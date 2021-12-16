import numpy
from typing import Callable
from functools import partial


_OPS: dict[str, Callable[[numpy.ndarray], numpy.ndarray]] = {
    "max": partial(numpy.max, axis=2),
    "mean": partial(numpy.mean, axis=2)
}


def project_block(
    block: numpy.ndarray,
    op: str
) -> numpy.ndarray:
    return _OPS[op](block)
