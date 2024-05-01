#!/bin/bash --login
# adapted from https://stackoverflow.com/questions/55123637/activate-conda-environment-in-docker/62674910#62674910
set -e

# activate conda environment and let the following process take over
conda activate flusion
exec "$@"
