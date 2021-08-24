"""
Module implementing feature extraction using CellProfiler modules

"""

import cellprofiler_core.pipeline
import cellprofiler_core.preferences
import cellprofiler.modules.measureimageintensity
import cellprofiler.modules.measuretexture
import pandas
import logging


def compute_measurements_on_partition(partition, *, modules, channels):

    logger = logging.getLogger(__name__)
    logger.debug('Entering compute_measurements_on_partition')

    cellprofiler_core.preferences.set_headless()
    pipeline = cellprofiler_core.pipeline.Pipeline()
    measurements = cellprofiler_core.measurement.Measurements()
    object_set = cellprofiler_core.object.ObjectSet()
    image_set_list = cellprofiler_core.image.ImageSetList()

    logger.debug('Starting iteration over images')

    idx = []
    for im in partition:
        idx.append(im["idx"])

        # populate image set '0' with current image
        # we don´t construct the full image_set_list to avoid copying
        image_set = image_set_list.get_image_set(0)
        for j in range(im["pixels"].shape[0]):
            cp_img = cellprofiler_core.image.Image(image=im["pixels"][j], mask=im["mask"][j])
            image_set.add(str(channels[j]), cp_img)

        for module in modules:
            workspace = cellprofiler_core.workspace.Workspace(
                pipeline,
                module,
                image_set,
                object_set,
                measurements,
                image_set_list,
            )
            module.run(workspace)
        measurements.next_image_set()

    logger.debug("All measurements collected")

    df = pandas.DataFrame(columns=[c for _, c, _ in measurements.get_measurement_columns()])
    for o, m, _ in measurements.get_measurement_columns():
        df[m] = measurements.get_all_measurements(o, m)
    df["idx"] = idx

    logger.debug("Measurements converted to pandas dataframe")

    return df.to_dict("records")


def extract_features(*, images, channels):

    modules = []
    module = cellprofiler.modules.measureimageintensity.MeasureImageIntensity()
    module.images_list.set_value([str(c) for c in channels])
    modules.append(module)

    module = cellprofiler.modules.measuretexture.MeasureTexture()
    module.images_or_objects.set_value(cellprofiler.modules.measuretexture.IO_IMAGES)
    module.add_scale()
    modules.append(module)

    return images.map_partitions(
        compute_measurements_on_partition,
        modules=modules,
        channels=channels
    ).to_dataframe().set_index("idx")
