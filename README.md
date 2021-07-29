# Scalable image processing

## Installation

We recommend using [mamba](https://github.com/mamba-org/mamba) and pip to install SCIP. 

1. Create a new python 3.8 environment: `mamba create -n scip python=3.8`
1. Activate environment: `conda activate scip`
1. Install dask: `mamba install dask`
1. Install CellProfiler:
    1. `pip install cellprofiler`
    1. or, if you get errors: `pip install -f https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-20.04/ cellprofiler`
1. Install SCIP: `pip install .` or `pip install -e .` for development
1. (Optional) Install SCIP development dependencies: `pip install -r requirements.txt`

## Usage

Usage: scip [OPTIONS] [PATHS]... [OUTPUT_DIRECTORY]

  Intro documentation

Options:
  -j, --n-workers INTEGER  how many workers are started in the dask cluster
  -p, --port INTEGER       dask dashboard port
  --debug                  sets logging level to debug
  --local / --no-local     deploy app to Dask LocalCluster, otherwise deploy
                           to dask-jobqueue PBSCluster
  --headless               If set, the program will never ask for user input
  --config FILE            Path to YAML config file
  --help                   Show this message and exit.
