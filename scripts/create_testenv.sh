#!/usr/bin/env bash

set -ex # fail on first error, print commands

command -v conda >/dev/null 2>&1 || {
  echo "Requires conda but it is not installed.  Run install_miniconda.sh." >&2;
  exit 1;
}

# if no python specified, use Travis version, or else 3.6
PYTHON_VERSION=${PYTHON_VERSION:-${TRAVIS_PYTHON_VERSION:-3.6}}


if [[ $* != *--global* ]]; then
    ENVNAME="testenv_${PYTHON_VERSION}"

    if conda env list | grep -q ${ENVNAME}
    then
        echo "Environment ${ENVNAME} already exists, keeping up to date"
    else
        echo "Creating environment ${ENVNAME}"
        conda create -n ${ENVNAME} --yes pip python=${PYTHON_VERSION}
    fi

    # Activate environment immediately
    source activate ${ENVNAME}

    if [ "$DOCKER_BUILD" = true ] ; then
        # Also add it to root bash settings to set default if used later

        echo "Creating .bashrc profile for docker image"
        echo "set conda_env=${ENVNAME}" > /root/activate_conda.sh
        echo "source activate ${ENVNAME}" >> /root/activate_conda.sh


    fi
fi


# Install PyMC4 dependencies
pip install --upgrade pip


#  Install editable using the setup.py
pip install --no-cache-dir -r requirements.txt
pip install --no-cache-dir -r requirements-dev.txt
