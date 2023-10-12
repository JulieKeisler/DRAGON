![alt text](docs/source/dragon_logo.jpeg)

[![Documentation Status](https://readthedocs.org/projects/dragon-tutorial/badge/?version=latest)](https://dragon-tutorial.readthedocs.io/en/latest/?badge=latest)
[![GitHub latest commit](https://badgen.net/github/last-commit/JulieKeisler/dragon/)](https://github.com/JulieKeisler/dragon/commit/)
![Maintainer](https://img.shields.io/badge/maintainer-J.Keisler-blue)


**DRAGON**, for **DiRected Acyclic Graphs OptimizatioN**, is an open source Python package for the optimization of *Deep Neural Networks Hyperparameters and Architecture* [[1]](#1). 
**DRAGON** is based on the package [Zellij](https://zellij.readthedocs.io/).

The distributed version requires a MPI library, such as [MPICH](https://www.mpich.org/)
or [Open MPI](https://www.open-mpi.org/).
It is based on [mpi4py](https://mpi4py.readthedocs.io/en/stable/intro.html#what-is-mpi).

See [documentation](https://dragon-tutorial.readthedocs.io/en/latest/).

## Dependencies ##

The following libraries are necessary:
* Zellij: [https://github.com/ThomasFirmin/zellij](https://github.com/ThomasFirmin/zellij)
* Pytorch: [https://pytorch.org/](https://pytorch.org/)
* MPI4py to use the kdistributed version: [https://mpi4py.readthedocs.io/en/stable/](https://mpi4py.readthedocs.io/en/stable/)
* GluonTS: [https://ts.gluon.ai/stable/](https://ts.gluon.ai/stable/) to compare with the [Monash Forecasting Repository](https://forecastingdata.org/) ([see github](https://github.com/rakshitha123/TSForecasting))

## Launch the optimization ##

You can try the optimization framework on any data from the [Monash Forecasting Archive](https://zenodo.org/communities/forecasting?page=1&size=20) whose config has been set in the config file: experiments/monash_archive/datasets_configs.py, by running:

`python template_optimization.py --dataset=dataset_name`

An MPI version is also available:

`mpiexec -np X python template_MPI_optimization.py --dataset=dataset_name`

## Contributors
### Design
* Julie Keisler: julie.keisler.rfo@gmail.com
  
## References
<a id="1">[1]</a>
Keisler, J., Talbi, E. G., Claudel, S., & Cabriel, G. (2023). An algorithmic framework for the optimization of deep neural networks architectures and hyperparameters. arXiv preprint arXiv:2303.12797.


