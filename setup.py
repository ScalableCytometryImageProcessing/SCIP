import setuptools

setuptools.setup(
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: LInux",
    ],
    name="scip",
    version="0.0.2",
    package_dir={"": "src"},
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
    ],
    entry_points={
        'console_scripts': [
            'scip=scip.main:cli'
        ]
    },
    packages=setuptools.find_packages(where="src"),
    python_requires="==3.8.*",
    package_data={
        'scip': ['logging.yml']
    }
)
