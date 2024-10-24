DRAGON
====================

**DRAGON**, for **DiRected Acyclic Graphs OptimizatioN**, is an open source Python package for the optimization of *Deep Neural Networks Hyperparameters and Architecture* [1]_. 
**DRAGON** is not a *no code* package, but you can get familiar with it quickly thanks to the `Quickstart <../Quickstart/quickstart.ipynb>`_ tutorial. 

Key Features
------------

- :ref:`A flexible seach space <../Search_space/index.rst>`
     - The search space based on Directed Acyclic Graphs (DAGs) where the nodes can be any **PyTorch** layer (custom or not) and the edges are the connections between them. 
     - The code to implement the DAGs-based search space was inspired by the `zellij <https://zellij.readthedocs.io/en/latest/>`__ package developed for hyperparameters optimization. 
     - **DRAGON** search space includes cell-based search spaces [4]_.
 
- Flexible optimization algorithms
     - The search algorithms defined in **DRAGON** are based on search operators used to modify elements of the search space (e.g., mutations, neighborhoods, crossover), which can be used to develop new search algorithms.
     - Efficient algorithms are also implemented in **DRAGON** such as the `Random Search <../Search_Algorithm/random_search.ipynb>`_, `Evolutionary Algorithm <../Search_Algorithm/ssea.ipynb>`_ [1]_, `Mutant UCB <../Search_Algorithm/mutant_ucb.ipynb>`_ [5]_, and `HyperBand <../Search_Algorithm/hyperband.ipynb>`_ [6]_.

- Applications to various tasks
     - The flexibility of **DRAGON** makes it usable for various applications.
     - For example: `image classification <../Applications/image.ipynb>`_, time series forecasting [1]_, `electricity consumption forecasting <../Applications/load_forecasting.ipynb>`_ [2]_, wind power forecasting [3]_ or tabular data.

- Easy parallelization over multiple GPUs
     - The distributed version requires a MPI library, such as `MPICH <https://www.mpich.org/>`_ or `Open MPI <https://www.open-mpi.org/>`_ and is based on `mpi4py <https://mpi4py.readthedocs.io/en/stable/intro.html#what-is-mpi>`_.

Basic Concepts
------------

- The **Search Space** is a mix-variable search space. Numerical, categorical and graph objects may be jointly optimized. Each object is associated with a **variable**, which defines what values an object can take.
- Base on this search space, several **Search Operators** are defined, showing how the objects can be manipulate to find the neighboring values.

Install DRAGON
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
