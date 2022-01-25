.. SCIP documentation master file, created by
   sphinx-quickstart on Mon Jan  3 11:50:35 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Scalable Cytometry Image Processing
===================================

Installation
------------

SCIP requires an MPI implementation to be present on the path. If no implementation is present,
we recommend installing mpi4py in a conda environment.

We recommend installing in a conda environment using `mamba <https://github.com/mamba-org/mamba>`_.

1. Create a new python environment: ``mamba create -n scip python=3.{8,9}``
2. Activate environment: ``conda activate scip``
3. Install mpi4py: ``mamba install -c conda-forge mpi4py``
4. Install SCIP: ``pip install .`` or ``pip install -e .`` for development
5. (Optional) Install development dependencies: ``pip install -r requirements.txt``

Usage
-----

Input
^^^^^
SCIP handles three types of input:

- one or more directories of (multiframe) TIFF images,
- one or more Carl Zeiss Image (CZI) images, and
- one or more `zarr files <https://zarr.readthedocs.io/en/stable/>`_.

Zarr
""""
Imaging flow cytometry datasets typically contain a vast amount of small image files, which are
inefficient and cumbersome to store and load. Zarr compatibility was added to SCIP so that these
images can be grouped into a couple of zarr files, which are much easier to handle.

The images are stored in the zarr file using the VLenArray object codec, which supports arrays of
varying length. This is described in `the zarr documentation <https://zarr.readthedocs.io/en/stable/tutorial.html?highlight=ragged#ragged-arrays>`_.
This way the images don't have to be cropped or padded to account
for their varying X and Y dimensions. The images do have to be flattened as the coded only
supports 1D arrays. SCIP reshapes the images upon loading using the shape attribute described below.

The zarr file should have two attributes:

1. "object_number": A list containing a unique identifier for each image.
2. "shape": A list containing the unflattened shape of each image.

SCIP assumes that the length of the lists in both attributes is equal to the length
of the zarr array.

API
---
.. toctree::
   :maxdepth: 2

   scip

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
