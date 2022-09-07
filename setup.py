import setuptools
import versioneer

setuptools.setup(
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Image Processing"
    ],
    name="scip",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    install_requires=[
        'numpy',
        'Pillow',
        'dask>=2022.01.1',
        'dask-ml',
        'click',
        'scikit-image',
        'pyyaml',
        'graphviz',
        'pyarrow',
        'umap-learn',
        'bokeh',
        'zarr',
        'fastparquet',
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
