# EvoDagsAutoDL #

Code for peprint paper -- An algorithmic framework for the optimization of deep neural networks architectures and hyperparameters

If you use this code please cite:

## Dependencies ##

The following libraries are necessary:
* Zellij: [https://github.com/ThomasFirmin/zellij](https://github.com/ThomasFirmin/zellij)
* Pytorch: [https://pytorch.org/](https://pytorch.org/)
* MPI4py to use the parallelization: [https://mpi4py.readthedocs.io/en/stable/](https://mpi4py.readthedocs.io/en/stable/)
* GluonTS: [https://ts.gluon.ai/stable/](https://ts.gluon.ai/stable/) to compare with the [Monash Forecasting Repository](https://forecastingdata.org/) ([https://github.com/rakshitha123/TSForecasting](https://github.com/rakshitha123/TSForecasting))

## Launch the optimization ##

You can try the optimization framework on any data from the [Monash Forecasting Archive](https://zenodo.org/communities/forecasting?page=1&size=20) whose config has been set in the config file: experiments/monash_archive/datasets_configs.py, by running:

`python template_optimization.py --dataset=dataset_name`

An MPI version is also available:

`mpiexec -np X python template_MPI_optimization.py --dataset=dataset_name`


