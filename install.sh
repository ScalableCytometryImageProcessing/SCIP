#!/bin/bash

INSTALLER=$1
ENV_NAME=scip

source $CONDA_PREFIX/etc/profile.d/conda.sh

$INSTALLER create -y -n $ENV_NAME python=3.8 &&\
conda activate $ENV_NAME &&\
$INSTALLER install -y dask &&\
pip install -f https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-18.04/ cellprofiler &&\
pip install -e . &&\
pip install -r requirements.txt
