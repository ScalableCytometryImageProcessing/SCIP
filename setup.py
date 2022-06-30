import setuptools
import versioneer

setuptools.setup(
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Linux",
    ],
    name="scip",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    install_requires=[
        'numpy',
        'Pillow',
        'dask',
        'dask-ml',
        'click',
        'scikit-image',
        'pyyaml',
        'graphviz',
        'pyarrow',
        'umap-learn',
        'bokeh',
        'zarr',
        'fastparquet'
    ],
    entry_points={
        'console_scripts': [
            'scip=scip.main:cli'
        ]
    },
    extras_require={
        "mpi": ['dask_mpi', 'mpi4py'],
        "czi": ['cellpose', 'aicsimageio', 'aicspylibczi'],
        "jobqueue": [
            "dask_jobqueue @ git+https://github.com/ScalableImagingPipeline/dask-jobqueue.git"]
    },
    python_requires=">=3.8",
    package_data={
        'scip': ['utils/logging.yml']
    }
)
