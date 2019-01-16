#! /bin/bash
SRC_DIR=${SRC_DIR:-`pwd`}

# Build container for use of testing
if [[ $* == *--build* ]]; then
    echo "Building Docker Image"
    docker build \
        -t pymc4 \
        -f $SRC_DIR/scripts/Dockerfile \
        --build-arg SRC_DIR=. $SRC_DIR \
        --rm
fi

if [[ $* == *--clear_cache* ]]; then
    echo "Removing cached files"
    find -type d -name __pycache__ -exec rm -rf {} +
fi

if [[ $* == *--test* ]]; then
    echo "Testing PyMC4"
    docker run --mount type=bind,source="$(pwd)",target=/opt/pymc4/ pymc4:latest bash -c \
                                      "pytest -v pymc4/tests/ --cov=pymc4/"
fi
