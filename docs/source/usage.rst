
Usage
-----

SCIP can be used as a command line interface (CLI) or its modules can be imported in a custom script. See the package API documentation for the latter option.

The CLI runs a workflow that loads images and performs projection, illumination correction, segmentation, masking, and feature extraction. The CLI can be configured via options passed on the command line (for runtime configuration) and via a YAML config file (for pipeline configuration). For an overview of the command line options, run ``scip --help``.

The YAML config file has the following specification:

.. code-block:: yaml

    load:
        format: {tiff, czi, multiframe_tiff, zarr}
        channels: []
        channel_names:
            - channel 1
            - channel 2
            - ...
        kwargs:
            regex: "^(?P<key_to_extract_from_filename>[0-9]+)$"
            scenes: {[], scene_name, regex_pattern} # only for czi loader
    project:
        method: {op}
        settings:
            op: {median, max}
    illumination_correction:
        method: {jones_2006}
        key: grouping_key
        export: {true, false}
        settings:
            median_filter_size: integer
            downscale: {none, integer}
    segment:
        method: {cellpose}
        export: {true, false}
        settings:
            cell_diameter: {none, integer}
            dapi_channel_index: index to channels list
            parent_channel_index: index to channels list
            substract:
                left_index: index to channels list
                right_index: index to channels list
                for_index: index to channels list
    mask:
        combined_indices: [0, 5, 8]
        methods:
            - method: "threshold"
              name: "threshold-name"
              bbox_channel_index: index to channels list
              export: {true, false}
              kwargs:
                  smooth: [0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0.5, 0.5, 0.5]
            - method: "spot"
              name: "spot-name"
              bbox_channel_index: index to channels list
              export: {true, false}
              kwargs:
                  spotsize: 5
    # with masks
    feature_extraction:
        threshold-name: ["shape", "intensity", "bbox", "texture"]
        spot-name: ["shape", "intensity", "bbox", "texture"]
    # without masks
    feature_extraction: ["shape", "intensity", "bbox", "texture"]
    export:
        format: {parquet, anndata}
        filename: name

You can use the example above as a start for your configuration.

Note that all top-level keys have to be present. You can disable a step by only providing the top-level key, like so:

.. code-block:: yaml

    mask:
