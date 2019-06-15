## Optical Pooled Screens

Example _in situ_ sequencing-by-synthesis data (in `example_data/`) and analysis code (in `ops/`) for the manuscript *Pooled optical screens in human cells*, available on [Biorxiv](https://www.biorxiv.org/content/early/2018/08/02/383943).


### Installation (OSX)

Download the OpticalPooledScreens directory (e.g., on Github use the green "Clone or download" button, then "Download ZIP").

In Terminal, go to the OpticalPooledScreens project directory and create a Python 3 virtual environment using a command like:

```bash
python3 -m venv venv
```

If the python3 command isn't available, you might need to specify the full path. E.g., if [Miniconda](https://conda.io/miniconda.html) is installed in the home directory:

```bash
~/miniconda3/bin/python -m venv venv
```

This creates a virtual environment called `venv` for project-specific resources. The commands in `install.sh` add required packages to the virtual environment:

```bash
sh install.sh
```

The `ops` package is installed with `pip install -e`, so the source code in the `ops/` directory can be modified in place. The package is compatible with  Python 2.7 and Python 3.6, however results may not be numerically identical.

## Running example code

Once installed, activate the virtual environment from the project directory:

```bash
source venv/bin/activate
```

You can then launch a project-specific notebook server:


```bash
jupyter notebook
```

The notebook `ops_python.ipynb` demonstrates step-by-step analysis using the high-level functions in `ops.firesnake.Snake`. The analysis pipeline can also be run using [snakemake](https://snakemake.readthedocs.io/en/stable/) (after activating the virtual environment):


```bash
cd example_data
snakemake -s Snakefile_20180707_201
```
