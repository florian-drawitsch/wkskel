# wkskel

A library for working with high-level skeleton representations of webKnossos nml files.

## Package installation with pip (local)

To install the package locally run pip install in the wkskel main folder

``` bash
pip install .
```

or via a symlink that makes any changes in the repository directly available to
all code on the system via
``` bash
pip install -e .
```

## Setting up conda (development) enviroment

To create a conda environment for wkskel run the following command in the wkskel 
main folder

``` bash
conda env create -f environment.yml
```

Then, to activate the environment run
``` bash
conda activate wkskel
```