
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