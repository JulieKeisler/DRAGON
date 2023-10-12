DRAGON Documentation
====================

**DRAGON**, for DiRected Acyclic Graphs OptimizatioN, is an open source Python package for the optimization of *Deep Neural Networks Hyperparameters and Architecture* [1]_. 
**DRAGON** is based on the package `Zellij <https://zellij.readthedocs.io/>`__.

The parallelized version requires a MPI library, such as `MPICH <https://www.mpich.org/>`__
or `Open MPI <https://www.open-mpi.org/>`__.
It is based on `mpi4py <https://mpi4py.readthedocs.io/en/stable/intro.html#what-is-mpi>`__.

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

Contributors
------------
* Julie Keisler: julie.keisler.rfo@gmail.com
References
----------
.. [1] Keisler, J., Talbi, E. G., Claudel, S., & Cabriel, G. (2023). An algorithmic framework for the optimization of deep neural networks architectures and hyperparameters. arXiv preprint arXiv:2303.12797.
