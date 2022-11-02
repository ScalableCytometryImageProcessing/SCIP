
Installation
============

SCIP can be installed from the PyPi repository,
or from source.

We recommend installing SCIP in a conda environment
using `mambaforge <https://github.com/conda-forge/miniforge>`_.

Prerequisites
-------------

If you want to use GPU-accelerated Cellpose segmentation,
make sure to install a compatible cudatoolkit. The version
you need depends on which GPU model you want to use.

If you want to run SCIP on a `dask-mpi <http://mpi.dask.org/en/latest/>`_ cluster,
an MPI implementation must be available. We recommend installing mpich.

.. code-block::

    mamba install -c conda-forge mpich-mpicc

PyPi
----

SCIP can be installed from the PyPi repository. The base
installation can be installed as follows:

.. code-block::

    pip install scip

There are also optional dependencies:
* mpi: Add support for running SCIP using a `dask-mpi <http://mpi.dask.org/en/latest/>`_ cluster,
* cellpose: Add support for `cellpose <https://www.cellpose.org/>`_ functionality,
* czi: Add support for reading Carl Zeiss Image files, and
* dev: Installing extra package related to development.

These can be installed using pip's bracket notation:

.. code-block::

    pip install scip[mpi]
    pip install scip[cellpose]
    pip install scip[cellpose,mpi,czi]

From source
-----------

To install SCIP from source:

1. Download the latest release from Github, or clone the repository.
2. Optionally extract the release.
3. Enter the repository directory.
4. Run ``pip install .`` if you only need default functionality.
5. (Optional) Run ``pip install .[extra]`` for extra functionality (refer to the list above
   for available extras)
