import setuptools
import versioneer

setuptools.setup(
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: LInux",
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
        'dask_jobqueue @ git+https://github.com/ScalableImagingPipeline/dask-jobqueue.git',
        'scikit-image',
        'pyyaml',
        'graphviz',
        'pyarrow',
        'umap-learn',
        'dask_mpi',
        'aicsimageio[czi]',
        'centrosome',
        'mpi4py',
        'bokeh',
        'zarr',
        'fastparquet',
        'cellpose'
    ],
    entry_points={
        'console_scripts': [
            'scip=scip.main:cli'
        ]
    },
    python_requires=">=3.8",
    package_data={
        'scip': ['logging.yml']
    }
)
