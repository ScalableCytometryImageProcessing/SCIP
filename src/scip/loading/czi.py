from aicsimageio import AICSImage
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed, expand_labels
from skimage import feature, measure
from scipy.ndimage import distance_transform_edt

import numpy
import dask.bag
import dask.dataframe
import dask.array
import dask
from centrosome import radial_power_spectrum
import scipy.linalg
from skimage.measure import regionprops
import pandas


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


def select_focused_plane(block):
    scores = numpy.empty(shape=block.shape[:3], dtype=float)
    for m, c, z in numpy.ndindex(block.shape[:3]):
        scores[m, c, z] = compute_powerslope(block[m, c, z])[0]
    indices = numpy.squeeze(scores.argmax(axis=2))
    return numpy.vstack([block[:, i, j] for i, j in enumerate(indices)])[numpy.newaxis]


@dask.delayed
def segment_block(block, *, idx, cell_diameter, dapi_channel):

    plane = block[0, dapi_channel]

    t = threshold_otsu(plane)
    cells = plane > t
    distance = distance_transform_edt(cells)

    local_max_coords = feature.peak_local_max(distance, min_distance=cell_diameter)
    local_max_mask = numpy.zeros(distance.shape, dtype=bool)
    local_max_mask[tuple(local_max_coords.T)] = True
    markers = measure.label(local_max_mask)

    segmented_cells = watershed(-distance, markers, mask=cells)
    segmented_cells = expand_labels(segmented_cells, distance=cell_diameter*0.25)

    events = []
    props = regionprops(segmented_cells)
    for i, prop in enumerate(props):
        bbox = prop.bbox
        events.append(dict(
            pixels=block[0, :, bbox[0]: bbox[2], bbox[1]:bbox[3]],
            mask=numpy.repeat(prop.image[numpy.newaxis], block.shape[1], axis=0),
            idx=f"{idx}_{i}",
            group=idx,
            bbox=tuple(bbox)
        ))

    return events


@dask.delayed
def meta_from_delayed(events, path, tile, scene):
    df = pandas.DataFrame.from_records([
        dict(idx=event["idx"], path=path, tile=tile, scene=scene) for event in events
    ]).set_index("idx")
    df.columns = [f"meta_{c}" for c in df.columns]
    return df


def bag_from_directory(*, path, idx, channels, partition_size, dapi_channel, cell_diameter, scenes):

    im = AICSImage(path, reconstruct_mosaic=False, chunk_dims=["Z", "C", "X", "Y"])

    data = []
    scenes_meta = []
    for scene in scenes:
        im.set_scene(scene)
        data.append(im.get_image_dask_data("MCZXY", T=0, C=channels))
        scenes_meta.extend([scene]*data[-1].numblocks[0])
    data = dask.array.concatenate(data)

    data = data.map_blocks(
        select_focused_plane, 
        drop_axis=2, 
        dtype=data.dtype, 
        meta=numpy.array((), dtype=data.dtype)
    )

    delayed_blocks = data.to_delayed().flatten()

    cells = []
    meta = []
    for tile, block in enumerate(delayed_blocks):
        scene = scenes_meta[block.key[1]]
        cells.append(segment_block(
            block, 
            idx=f"{idx}_{scene}_{tile}",
            cell_diameter=cell_diameter,
            dapi_channel=dapi_channel
        ))
        meta.append(
            meta_from_delayed(cells[-1], path=path, tile=tile, scene=scene))

    return dask.bag.from_delayed(cells), dask.dataframe.from_delayed(meta)
