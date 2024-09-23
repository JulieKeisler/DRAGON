DRAGON Documentation
====================

**DRAGON**, for **DiRected Acyclic Graphs OptimizatioN**, is an open source Python package for the optimization of *Deep Neural Networks Hyperparameters and Architecture* ref1_. The Deep Neural Networks are represented using Directed Acyclic Graphs (DAGs), where the nodes can be any **PyTorch** layer (custom or not) and the edges are the connections between them. **DRAGON** is not a *no code* package, but you can get familiar with it quickly thanks to the `Quickstart <../Quickstart/quickstart.ipynb>`_ tutorial. 
On the other hand, the package allows you to modulate the search space at will for a given application, such as `image classification <../Applications/image.ipynb>`_, time series forecasting ref1_, `electricity consumption forecasting <../Applications/load_forecasting.ipynb>`_ [2]_, wind power forecasting [3]_, tabular data, etc.

The code to implement the DAGs-based search space was inspired by the `zellij <https://zellij.readthedocs.io/en/latest/>`__ package developed for hyperparameters optimization. The  :ref:`_search_space` section describes its implementation and advanced features to further modulate the search space. In particular, we show that the **DRAGON** search space includes cell-based search spaces [4]_.

The Search Algorithm section introduces the search operators used to modify elements of the search space (e.g., mutations, neighborhoods, crossover). It also describes the algorithms already implemented in **DRAGON**: :`Random Search <../Search_Algorithm/random_search.ipynb>`_, `Evolutionary Algorithm <../Search_Algorithm/ssea.ipynb`_ ref1_, `Mutant UCB <../Search_Algorithm/mutant_ucb.ipynb`_ [5]_, and `HyperBand <../Search_Algorithm/hyperband.ipynb`_ [6]_.

The distributed version requires a MPI library, such as `MPICH <https://www.mpich.org/>`_
or `Open MPI <https://www.open-mpi.org/>`_.
It is based on `mpi4py <https://mpi4py.readthedocs.io/en/stable/intro.html#what-is-mpi>`_.

Install DRAGON
--------------

Basic version
^^^^^^^^^^^^^

After cloning the git repository, install **DRAGON**, using:

.. code-block:: bash

     pip install -e dragon

Distributed version
^^^^^^^^^^^^^

If you plan on using the distributed version, you have to install the mpi4py package:

.. code-block:: bash

     pip install mpi4py

Dependencies
------------

* **Python** >=3.9
* `numpy <https://numpy.org/>`__>=1.21.4
* `DEAP <https://deap.readthedocs.io/en/master/>`__>=1.3.1
* `botorch <https://botorch.org/>`__>=0.6.3.1
* `gluonts <https://ts.gluon.ai/stable/>`__>=0.11.3
* `gpytorch <https://gpytorch.ai/>`__>=1.6.0
* `graphviz <https://graphviz.org/>`__>=0.8.4
* `pandas <https://pandas.pydata.org/>`__>=1.3.4
* `enlighten <https://python-enlighten.readthedocs.io/en/stable/>`__>=1.10.2
* [mpi]: `mpi4py <https://mpi4py.readthedocs.io/en/stable/>`__>=3.1.2

Contributors
------------
* Julie Keisler: julie.keisler.rfo@gmail.com
References
----------
.. _ref1:
     Keisler, J., Talbi, E. G., Claudel, S., & Cabriel, G. (2024). An algorithmic framework for the optimization of deep neural networks architectures and hyperparameters. *Journal of Machine Learning Research*, 25(201), 1-33.
.. [2] Keisler, J., Claudel, S., Cabriel, G., & Brégère, M. (2024). Automated Deep Learning for Load Forecasting. *International Conference on Automated Machine Learning*.
.. [3] Keisler, J., & Naour, E. L. (2024). WindDragon: Enhancing wind power forecasting with Automated Deep Learning. Workshop paper at *Tackling Climate Change with Machine Learning*, *International Conference on Learning Representations*.
.. [4] Elsken, T., Metzen, J. H., & Hutter, F. (2019). Neural architecture search: A survey. *Journal of Machine Learning Research*, 20(55), 1-21.
.. [5] Brégère, M., & Keisler, J. (2024). A Bandit Approach with Evolutionary Operators for Model Selection.
.. [6] Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A., & Talwalkar, A. (2018). Hyperband: A novel bandit-based approach to hyperparameter optimization. *Journal of Machine Learning Research*, 18(185), 1-52.
