import numpy
from centrosome import radial_power_spectrum
import scipy.linalg


def compute_powerslope(pixel_data):
    radii, magnitude, power = radial_power_spectrum.rps(pixel_data)
    if sum(magnitude) > 0 and len(numpy.unique(pixel_data)) > 1:
        valid = magnitude > 0
        radii = radii[valid].reshape((-1, 1))
        power = power[valid].reshape((-1, 1))
        if radii.shape[0] > 1:
            idx = numpy.isfinite(numpy.log(power))
            powerslope = scipy.linalg.basic.lstsq(
                numpy.hstack(
                    (
                        numpy.log(radii)[idx][:, numpy.newaxis],
                        numpy.ones(radii.shape)[idx][:, numpy.newaxis],
                    )
                ),
                numpy.log(power)[idx][:, numpy.newaxis],
            )[0][0]
        else:
            powerslope = 0
    else:
        powerslope = 0
    return powerslope


def project_block(block: numpy.ndarray) -> numpy.ndarray:
    scores = numpy.empty(shape=block.shape[:3], dtype=float)
    for m, c, z in numpy.ndindex(block.shape[:3]):
        slope = compute_powerslope(block[m, c, z])
        if hasattr(slope, "shape"):
            scores[m, c, z] = slope[0]
        else:
            scores[m, c, z] = slope

    indices = numpy.squeeze(scores.argmax(axis=2))
    return numpy.vstack([block[:, i, j] for i, j in enumerate(indices)])[numpy.newaxis]
