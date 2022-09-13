import setuptools
import versioneer

setuptools.setup(
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Image Processing"
    ],
    name="scip",
    license="GPL-2.0-or-later",
    maintainer="Maxim Lippeveld",
    author_email="maxim.lippeveld@ugent.be",
    description="Scalable Cytometry Image Processing",
    long_description="Scalable Cytometry Image Processing (SCIP) is an open-source tool that implements an image processing pipeline on top of Dask, a distributed computing framework written in Python. SCIP performs normalization, image segmentation and masking, and feature extraction.",
    url="https://github.com/ScalableCytometryImageProcessing/SCIP",
    project_urls={
        'Documentation': "https://scalable-cytometry-image-processing.readthedocs.io/en/latest/",
        'Source': "https://github.com/ScalableCytometryImageProcessing/SCIP",
        'Tracker': "https://github.com/ScalableCytometryImageProcessing/SCIP/issues"
    },
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    install_requires=[
        'numpy',
        'Pillow',
        'dask>=2022.01.1',
        'click',
        'scikit-image',
        'pyyaml',
        'graphviz',
        'zarr',
        'pyarrow',
        'anndata'
    ],
    entry_points={
        'console_scripts': [
            'scip=scip.main:cli'
        ]
    },
    extras_require={
        "mpi": ['dask_mpi', 'mpi4py'],
        "czi": ['cellpose', 'aicsimageio', 'aicspylibczi'],
        "jobqueue": ["dask-jobqueue"]
    },
    python_requires=">=3.8",
    package_data={
        'scip': ['utils/logging.yml']
    }
)
