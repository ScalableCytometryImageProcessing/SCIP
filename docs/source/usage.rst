
Usage
-----

SCIP can be used as a command line interface (CLI) or its modules can be imported in a custom script. See the package API documentation for the latter option.

The CLI runs a workflow that loads images and performs segmentation, masking, filtering and feature extraction. The CLI can be configured via options passed on the command line (for runtime configuration) and via a YAML config file (for pipeline configuration). For an overview of the command line options, run ``scip --help``.

The YAML config file has the following specification:

.. code-block::
