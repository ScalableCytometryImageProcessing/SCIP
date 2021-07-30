#!/bin/bash

source $CONDA_PREFIX/etc/profile.d/conda.sh

mamba create -y -n scip python=3.8 &&\
conda activate scip &&\
mamba install -y dask &&\
pip install -f https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-20.04/ cellprofiler &&\
pip install -e . &&\
pip install -r requirements.txt
