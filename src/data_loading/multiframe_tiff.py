import dask
from PIL import Image
from dask.delayed import Delayed
import numpy
from pathlib import Path

@dask.delayed
def load_image(p: str) -> dict[numpy.ndarray, str]:
    im = Image.open(p)
    arr = numpy.empty(shape=(im.n_frames, im.height, im.width), dtype=float)
    for i in range(im.n_frames):
        im.seek(i)
        arr[i] = numpy.array(im)
    return dict(pixels=arr, path=p)

def from_directory(path: str) -> list[Delayed]:
    """
    Construct delayed ops for all tiffs in a directory

    path (str): Directory to find tiffs

    """

    ops = []
    for p in Path(path).glob("**/*.tiff"):
        ops.append(load_image(str(p)))

    return ops

