import setuptools

setuptools.setup(
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: LInux",
    ],
    name="pipelineModules",
    version="0.0.1",
    package_dir={"": "src"},
    install_requires=[
        'numpy',
        'Pillow',
        'dask',
        'dask_jobqueue',
        'click'
    ],
    entry_points = {
        'console_scripts': [
            'sip=main:cli'
        ]
    },
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7",
)
