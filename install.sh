#!/usr/bin/env bash

# #Installation
#
# Download the OpticalPooledScreens directory using the green "Clone or download" button on Github.
# 
# In Terminal, go to the OpticalPooledScreens project directory and create a Python 3 virtual environment using a command like:
#
#    python3 -m venv venv
# 
# If the python3 command isn't available, you might need to specify the full path.
# E.g., if miniconda3 is installed in the home directory:
# 
#    ~/miniconda3/bin/python -m venv venv
#
# This creates a virtual environment called venv for project-specific resources. 
# The script below installs required packages into the virtual environment.
#
# To reinstall, just delete the venv directory, re-create it as above, and re-run
# this script.
#
# Once installed, activate the virtual environment from the project directory:
#
#   source venv/bin/activate
#
# You can then launch a project-specific notebook server:
#
#   jupyter notebook
#
# Or analyze the data using snakemake:
#
#   cd example_data
#   snakemake -s Snakefile_20180707_201
#


source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements_skfmm.txt
# link ops package instead of copying
# jupyter and snakemake will import code from .py files in the ops/ directory
pip install -e .

