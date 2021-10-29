# SCIP: Scalable Image Processing

![main workflow badge](https://github.com/ScalableImagingPipeline/dask-pipeline/actions/workflows/main.yml/badge.svg)

## Installation

We recommend using [mamba](https://github.com/mamba-org/mamba) and pip to install SCIP.

1. Create a new python 3.8 environment: `mamba create -n scip python=3.8`
1. Activate environment: `conda activate scip`
1. Install mpi4py: `mamba install -c conda-forge mpi4py`
1. Install SCIP: `pip install .` or `pip install -e .` for development
1. (Optional) Install SCIP development dependencies: `pip install -r requirements.txt`

## Usage

SCIP is called from the command line using the `scip` command. The help output is shown below.

```
Usage: scip [OPTIONS] OUTPUT CONFIG [PATHS]...

  Intro documentation

Options:
  -d, --port INTEGER              dask dashboard port
  --debug                         sets logging level to debug
  --mode [local|jobqueue|mpi]     In which mode to run Dask
  -j, --n-workers INTEGER         Number of workers in the LocalCluster or per
                                  node
  -n, --n-nodes INTEGER           Number of nodes started
  -c, --n-cores INTEGER RANGE     Number of cores available per node in the
                                  cluster  [x>=1]
  -t, --n-threads INTEGER RANGE   Number of threads per worker process  [x>=1]
  -m, --memory INTEGER RANGE      Amount of memory available per node in the
                                  cluster  [x>=1]
  -w, --walltime TEXT             Expected required walltime for the job to
                                  finish
  -p, --project TEXT              Project name for HPC cluster
  -e, --job-extra TEXT            Extra arguments for job submission
  --headless                      If set, the program will never ask for user
                                  input
  -s, --partition-size INTEGER RANGE
                                  Set partition size  [x>=1]
  --timing FILE
  --report / --no-report
  -l, --local-directory DIRECTORY
  --help                          Show this message and exit.

```

### mode

SCIP can run in three different modes: local, mpi or jobqueue.
 - In local mode, SCIP spins up a Dask `LocalCluster`. This mode can be used for execution on a laptop or desktop, for example.
 - In mpi mode, Dask workers and the scheduler are setup using the `dask-mpi` package. This is the ideal mode for use on high performance clusters. Note that for this mode SCIP should be run using `mpirun` or `mpiexec`.
 - Finally, in jobqueue mode, a PBSCluster is set up using the `dask-jobqueue` package. This mode can also be used for execution on a high performance cluster if MPI is not available, for instance.

### OUTPUT, CONFIG and PATHS

`OUTPUT` should be passed as a path to a directory where SCIP can write its output. `CONFIG` should be passed as the path to a YAML-file containing the configuration for the execution of SCIP. The files [scip_czi.yml](scip_czi.yml), [scip_multi_tiff.yml](scip_multi_tiff.yml) and [scip_tiff.yml](scip_tiff.yml) contain example configurations. `PATHS` should point to one or more images or paths containing images.
