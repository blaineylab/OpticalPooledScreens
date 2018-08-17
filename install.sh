#!/usr/bin/env bash

# to reinstall, just delete the env directory

if [ ! -d "venv" ]; then
	virtualenv -p python3 env
	source venv/bin/activate
	pip install -r requirements.txt
    # link ops package instead of copying
    # jupyter and snakemake will import files from ops/ directory
    pip install -e ops
else
    echo "already installed, activating virtual environment (venv)"
    source env/bin/activate
fi

