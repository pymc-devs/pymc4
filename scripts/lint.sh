#! /bin/bash

set -ex # fail on first error, print commands

SRC_DIR=${SRC_DIR:-$(pwd)}

echo "Skipping documentation check. Re-enabling this would be a helpful contribution!"
# echo "Checking documentation..."
# python -m pydocstyle --convention=numpy "${SRC_DIR}"/pymc4/
echo "Success!"

echo "Checking code style with black..."
python -m black -l 100 --check "${SRC_DIR}"/pymc4/ "${SRC_DIR}"/tests/
echo "Success!"

echo "Type checking with mypy..."
python -m mypy --ignore-missing-imports "${SRC_DIR}"/pymc4/

echo "Checking code style with pylint..."
python -m pylint "${SRC_DIR}"/pymc4/ "${SRC_DIR}"/tests/
echo "Success!"
