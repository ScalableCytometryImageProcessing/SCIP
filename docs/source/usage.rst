
Usage
-----

SCIP can be used as a command line interface (CLI) or its modules can be imported in a custom script. See the API documentation for the latter option.

The CLI runs a workflow that loads images and performs segmentation, masking, filtering and feature extraction. The CLI can be configured via options passed on the command line (for runtime configuration) and via a YAML config file (for pipeline configuration). For an overview of the command line options, run ``scip --help``.

The YAML config file has the following specification:

.. code-block::

      loading:
         format: czi, multiframe_tiff, tiff, zarr  # loader module to use
         channels: [0]  # list of channel indexes, e.g. index of frame in multiframe tiff
         channel_names:  # list of channel names
            - "channel_0"
         kwargs:  # keyword arguments passed to loader function
            regex:  # regex pattern, can contain match groups for extracting metadata from filename

            # CellPose specific arguments
            segment_method: cellpose
            segment_kw:
               cell_diameter: 30  # expected cell diameter
               dapi_channel_index: 0
               main_channel_index: 2
            project_method: "op"
            project_kw:
               op: "max"

            # CZI loader specific arguments (nothing | list of scenes | regex pattern)
            scenes: | ["A2-A2", "A1-A1"] | "^A.*$"
      masking:
         method: watershed, threshold  # masking module to use
         bbox_channel_index: 0  # index of channel that shows cytoplasm
         export: false  # set to true to export masks
         combined_indices: [0]  # list of channel indexes of which masks will be combined
         kwargs:  # keyword arguments passed to masking function
            smooth: 1
            noisy_channels: []
      filter:
      normalization:
         lower: 0
         upper: 1
      feature_extraction:
         types: ["shape", "intensity", "bbox", "texture"]
      export:
         format: parquet
         filename: features