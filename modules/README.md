# Models, Architecture, and Training Code

[![License](https://img.shields.io/badge/license-CC--BY--4.0-green)](https://github.com/RikGhosh487/Open-Cluster/blob/main/LICENSE) ![Language](https://img.shields.io/badge/language-python-rgb(12%2C%2093%2C%20148)) [![Package](https://img.shields.io/badge/package-pytorch-blueviolet)](https://pytorch.org/)

This **directory** contains all the model files, the architecture code, the utility functions and the training code for the models. This entire directory is treated as a submodule and individual files can be executed from the main directory using
```
>>> python -m modules.<filename> [OPTIONS] ...
```

There are two models:
- **GAIA to SDSS**: takes 3 magnitudes from the filters used in the GAIA photometric system (`g`, `bp`, `rp`) and converts them to 5 magnitude values for the filters used in the SDSS PSF photometric system (`u`, `g`, `r`, `i`, `z`).

- **SDSS to GAIA**: takes 5 magnitudes from the filters used in the SDSS PSF photometric system (`u`, `g`, `r`, `i`, `z`) and converts them to 3 magnitude values for the filters used in the GAIA photometric system (`g`, `bp`, `rp`).

Both models are Multilayer Perceptrons (MLPs) that take either **3** or **5** input channels (depending on which model is being used) and propagates the channels to a max of **512** channels in the hidden layers, before converging to a final regressed value.

## Directory structure
```
- base (main)
    |
    |- data
        |- ...
    |- modules
        |- models.py
        |- train.py
        |- utils.py
        |- ...
    |- ...
```
- `__init__.py`: submodule initializer
- `gaia.th`: saved SDSS → GAIA model
- `models.py`: model architectures
- `sdss.th`: saved GAIA → SDSS model
- `train.py`: training code
- `utils.py`: utility functions and data loaders

