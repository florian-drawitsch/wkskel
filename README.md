# wkskel

*A python library for scientific analysis and manipulation of webKnossos skeleton tracings*.

![](./examples/wkskel.png)

In very broad sense, `wkskel` represents a python library for analysis and manipulation of *undirected acyclic graphs*, throughout the package referred to as __skeletons__. In a more applied sense, these skeletons typically represent neurite center-line reconstructions obtained manually using the in-browser neurite annotation tool [__webKnossos__](https://webknossos.brain.mpg.de/) or generated by automated methods. 

The `wkskel` library aims at providing the data structures and methods necessary to facilitate analysis, manipulation and visualization of such skeletons in a scientific workflow. Furthermore, it provides an input-output-interface to the __.nml__ format used by webKnossos.

## Package installation with pip (pypi)

To install the package via pip and the python package index:
```bash
pip install wkskel
```

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

For basic usage examples, see `examples`. A getting started guide in form of a jupyter notebook can be found [here](https://github.com/florian-drawitsch/wkskel/blob/master/examples/getting_started.ipynb)


