**Mammalian protein-coding genes exhibit widespread beneficial mutations that are not adaptive**\
Thibault Latrille, Julien Joseph, Diego Hartasánchez, Nicolas Salamin\
_PLoS Genetics_, 2024,\
[doi.org/10.1371/journal.pgen.1011536](https://doi.org/10.1371/journal.pgen.1011536)

**Compiled binaries and instructions for BayesCode are available at [github.com/ThibaultLatrille/bayescode](https://github.com/ThibaultLatrille/bayescode)**

# SelCoeff

This repository is meant to provide the necessary scripts and data to reproduce the figures shown in the manuscript.
The experiments can either run on a local computer or in a cluster configuration (slurm).

The experiments are meant to run on Linux/Unix/MacOS operating systems.

If problems and/or questions are encountered, feel free to [open issues](https://github.com/ThibaultLatrille/SelCoeff/issues).

## 0. Local copy
Clone the repository and `cd` to the dir.
```
git clone https://github.com/ThibaultLatrille/SelCoeff
cd SelCoeff
```

## 1. Installation

### General dependencies

Install python3 packages
```
sudo apt install -qq -y python3-dev python3-pip
pip3 install snakemake scipy numpy matplotlib pandas ete3 bio statsmodels --user
```

### Datasets
The datasets required to run the snakemake pipeline can be downloaded on Zenodo ([![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7878954.svg)](https://doi.org/10.5281/zenodo.7878954))

### Run global analysis contrasting polymorphism and divergence


In root folder run `snakemake`:
```
snakemake -j 32 -k
```

## 3. Add features or debug in the python scripts
You made modifications to one of the python script, a notebook, this README.md, or you added new features.
You wish this work benefits to all (futur) users of this repository?
Please, feel free to open a [pull-request](https://github.com/ThibaultLatrille8/SelCoeff/pulls)

## Licence

The MIT License (MIT)

Copyright (c) 2019 Thibault Latrille

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


