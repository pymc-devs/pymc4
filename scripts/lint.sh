#! /bin/bash

set -ex # fail on first error, print commands

SRC_DIR=${SRC_DIR:-`pwd`}

echo "Checking documentation with pydocstyle..."
python -m pydocstyle ${SRC_DIR}/pymc4/
echo "Success!"

echo "Checking code style with black..."
python -m black -l 100 --check pymc4/
echo "Success!"

echo "Checking code style with pylint..."
python -m pylint pymc4/
echo "Success!"
