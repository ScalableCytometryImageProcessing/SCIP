
Installation
------------

SCIP can be installed in three ways:
- with default functionality,
- with additional MPI support for running SCIP in MPI mode,
- and with additional CZI support for loading CZI files.

To install SCIP:

1. Download the latest release from Github, or clone the repository.
2. Enter the repository directory.
3. Run ``pip install .`` if you only need default functionality. Run ``pip install .[czi]`` for CZI support, and ``pip install .[mpi]`` for MPI support.
4.  (Optional) Install development dependencies: ``pip install -r requirements.txt``.

MPI support requires an MPI implementation to be present on the path. Please refer
to instructions for your operating system to install MPI. Another option is to install, mpi4py
in a conda environment.