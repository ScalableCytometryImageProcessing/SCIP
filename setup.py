import setuptools

setuptools.setup(
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: LInux",
    ],
    name="scip",
    version="0.0.1",
    package_dir={"": "src"},
    install_requires=[
        'numpy',
        'Pillow',
        'dask',
        'click',
        'dask_jobqueue @ git+https://github.com/ScalableImagingPipeline/dask-jobqueue.git',
        'scikit-image',
        'pyyaml',
        'graphviz',
        'cellprofiler',
        'pyarrow'
    ],
    entry_points={
        'console_scripts': [
            'scip=scip.main:cli'
        ]
    },
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.8",
)
