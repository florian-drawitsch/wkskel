# wkskel

A python library for scientific analysis and manipulation of webKnossos skeleton tracings.

## Package installation with pip (local)

To install the package locally run pip install in the wkskel main folder:
``` bash
pip install .
```

or via a symlink that makes any changes in the repository directly available to
all code on the system via:
``` bash
pip install -e .
```

## Setting up conda (development) enviroment

To create a conda environment for wkskel run the following command in the wkskel 
main folder:
``` bash
conda env create -f environment.yml
```

Then, to activate the environment run
``` bash
conda activate wkskel
```

## Getting started

For basic usage examples, see `examples`. A getting started guide in form of a jupyter notebook can be found [here](https://gitlab.mpcdf.mpg.de/connectomics/wkskel/blob/master/examples/getting_started.ipynb)