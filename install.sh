#!/usr/bin/env bash

# first, create the virtual environment using this command: python3 -m venv venv
# (the second venv/ refers to the folder where the virtual environment is made)
# (you will have to install the Python 3 virtualenv package if you don't have it)

# to reinstall, just delete the venv directory


if [ ! -d "venv" ]; then
	source venv/bin/activate
	pip install -r requirements.txt
	pip install -r requirements_skfmm.txt
    # link ops package instead of copying
    # jupyter and snakemake will import code from .py files in the ops/ directory
    pip install -e .
else
    echo "already installed, activating virtual environment (venv)"
    source venv/bin/activate
fi

