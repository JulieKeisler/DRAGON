Introduction
====================

**DRAGON**, for **DiRected Acyclic Graphs OptimizatioN**, is an open source Python package for the optimization of *Deep Neural Networks Hyperparameters and Architecture* [1]_.
**DRAGON** is not a *no code* package. It can be considered as a toolbox for the creation of Automated Deep Learning frameworks for various applications.

You can get familiar with it quickly thanks to the `Quickstart <../Quickstart/quickstart.ipynb>`_ tutorial. 

Structure
------------

- `Search Space <../Search_Space/index.rst>`_.
     - **DRAGON** provides tools to create custom search spaces. Various *Python* objects called *Variables* are available to encode different elements such as integers or arrays. These elements come from the `zellij <https://zellij.readthedocs.io/en/latest/>`__ package developed for hyperparameter optimisation.
     - Based on these elements, the search space based Directed Acyclic Graphs (DAGs) are proposed to encode the deep neural networks. The nodes can be any *PyTorch* layer (custom or not) and the edges are the connections between them. 

- `Search Operators <../Search_Operators/index.rst>`_.
     - Each *variable* can be given a *neighbor* attribute which is used to modify the current object. This function can be seen as a neighbourhood or mutation operator. The **DRAGON** package provides default mutations for each *Variable*, but the user is free to implement his own.
     - A crossover operator is also implemented, allowing both arrays and graph-like variables to be mixed.

- `Search Algorithms <../Search_Algorithm/index.rst>`_.
     - **DRAGON** provides the implementation of several search algorithms: the `Random Search <../Search_Algorithm/random_search.ipynb>`_, the `Evolutionary Algorithm <. ./Search_Algorithm/ssea.ipynb>`_ [1]_, `Mutant UCB <../Search_Algorithm/mutant_ucb.ipynb>`_ [5]_ and `Hyperband <../Search_Algorithm/hyperband.ipynb>`_ [6]_.
     - Mutant-UCB and the Evolutionary Algorithm use the *neighbor* attributes to modify the configurations. Other search algorithms such as local search or simulated annealing could be implemented in a similar way.
     - Each search algorithm comes with a storage system to keep RAM memory small and an optional distributed version on multiple processors. The distributed version requires an MPI library such as `MPICH <https://www.mpich.org/>`_ or `Open MPI <https://www.open-mpi.org/>`_ and is based on the `mpi4py package <https://mpi4py.readthedocs.io/en/stable/intro.html#what-is-mpi>`_.

- `Performance evaluation <../Applications/index.rst>`_.
     - Evaluating a configuration from a search space built with **DRAGON** is done by building the model (i.e. the neural networks) with the configuration elements. Then the model should be trained, evaluated and return a loss (the search algorithms minimise losses).
     - The process of building - training - evaluating a model based on a configuration depends on the applications and has to be implemented by the user.
     - Examples are given for `image classification <../Applications/image.ipynb>`_ with the package *skorch* and `electricity consumption forecasting <../Applications/load_forecasting.ipynb>`_ [2]_.


Installation
--------------

Basic version
^^^^^^^^^^^^^

After cloning the git repository, install **DRAGON**, using:

.. code-block:: bash

     pip install dragon-autodl==1.0

Distributed version
^^^^^^^^^^^^^

If you plan on using the distributed version, you have to install the mpi4py package:

.. code-block:: bash

     pip install mpi4py

Dependencies
------------

* **Python** >=3.9
* `numpy <https://numpy.org/>`__<2.0.0
* `torch <https://pytorch.org/>`__
* `graphviz <https://graphviz.org/>`__
* `pandas <https://pandas.pydata.org/>`__
* [mpi]: `mpi4py <https://mpi4py.readthedocs.io/en/stable/>`__
* [docs]: 
     * `openml <https://www.openml.org/>`__
     * `sklearn <https://scikit-learn.org>`__
     * `optuna <https://optuna.org/>`__
     * `matplotlib <https://matplotlib.org/>`__
     * `skorch <https://skorch.readthedocs.io/en/stable/>`__
     
Contributors
------------
* Julie Keisler: julie.keisler.rfo@gmail.com
References
----------
.. [1] Keisler, J., Talbi, E. G., Claudel, S., & Cabriel, G. (2024). An algorithmic framework for the optimization of deep neural networks architectures and hyperparameters. *Journal of Machine Learning Research*, 25(201), 1-33.
.. [2] Keisler, J., Claudel, S., Cabriel, G., & Brégère, M. (2024). Automated Deep Learning for Load Forecasting. *International Conference on Automated Machine Learning*.
.. [3] Keisler, J., & Naour, E. L. (2024). WindDragon: Enhancing wind power forecasting with Automated Deep Learning. Workshop paper at *Tackling Climate Change with Machine Learning*, *International Conference on Learning Representations*.
.. [4] Elsken, T., Metzen, J. H., & Hutter, F. (2019). Neural architecture search: A survey. *Journal of Machine Learning Research*, 20(55), 1-21.
.. [5] Brégère, M., & Keisler, J. (2024). A Bandit Approach with Evolutionary Operators for Model Selection.
.. [6] Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A., & Talwalkar, A. (2018). Hyperband: A novel bandit-based approach to hyperparameter optimization. *Journal of Machine Learning Research*, 18(185), 1-52.
