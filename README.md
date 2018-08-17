## Optical Pooled Screens

This repository holds analysis code (in `ops/`) and example _in situ_ sequencing-by-synthesis data (in `example_data/`) for the manuscript *Pooled optical screens in human cells* which is available on [Biorxiv](https://www.biorxiv.org/content/early/2018/08/02/383943).


### Installation

Download the OpticalPooledScreens directory using the green "Clone or download" button on Github.

In Terminal, go to the OpticalPooledScreens project directory and create a Python 3 virtual environment using a command like:

```bash
python3 -m venv venv
```

If the python3 command isn't available, you might need to specify the full path.
E.g., if miniconda3 is installed in the home directory:

```bash
~/miniconda3/bin/python -m venv venv
```

This creates a virtual environment called venv for project-specific resources. The script in `install.sh` installs required packages into the virtual environment:

```bash
sh install.sh
```

To reinstall, just delete the venv directory, re-create it as above, and re-run
this script.

Once installed, activate the virtual environment from the project directory:

```bash
source venv/bin/activate
```

You can then launch a project-specific notebook server:


```bash
jupyter notebook
```

The notebook `ops_python.ipynb` demonstrates step-by-step analysis using the high-level functions in `ops.firesnake.Snake`. The analysis pipeline can also be run using snakemake (after activating the virtual envrionment):


```bash
cd example_data
snakemake -s Snakefile_20180707_201
```

