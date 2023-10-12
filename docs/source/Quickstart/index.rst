==========
Quickstart
==========

In this tutorial, for computation time issues, we will use a small time series forecasting task.
(e.g. training a neural network is time consumming).
More applications here: :ref:`examples`.

Loading the dataset 
===================

.. code-block:: python

    from dragon.experiments.monash_archive.dataset import gluonts_dataset
    from dragon.experiments.monash_archive.datasets_config import m1_monthly_config

    train_ds, test_ds, config = gluonts_dataset(m1_monthly_config)

Defining the Loss Function
==========================

A :ref:`lf` consists in computing the prediction error of a pre-trained deep neural networks. The user is required to define a model, a training and a testing procedure.
Then, the package **Zellij** uses a wrapping function called :func:`zellij.core.Loss` to add features to the user defined function.


DNN definition
--------------

The class :func:`evodags.experiments.monash_archive.training.GluontsNet` handles the DNN creation, its training and testing procedure.

.. code-block:: python

    from evodags.experiments.monash_archive.training import GluontsNet

    model = GluontsNet(train_ds, test_ds, m1_monthly_config)

Loss function definition
------------------------

.. code-block:: python

  import numpy as np
  from zellij.core import Loss

  loss = Loss(verbose=False, save=True)(model.get_nn_forecast)


Defining the search space
=========================

To define a searchspace one need to define :ref:`var` and a :ref:`lf`.
Available :ref:`var` are:

* **Floats**: :class:`zellij.core.FloatVar` allows to model with upper and lower bounds a float decision variable. You can even change the sampler.
* **Integers**: :class:`zellij.core.IntVar` allows to model with upper and lower bounds a integer decision variable. You can even change the sampler.
* **Categorical**: :class:`zellij.core.CatVar` allows to model a categorical variable with a list of features.
* **Arrays**: :class:`zellij.core.ArrayVar` allows to model an array of :ref:`var`.
* **AdjMatrix**: :class: `evodags.search_space.dags import AdjMatrixVariable` allow to model a DNN. One needs to specify its operations as in the following example.


.. code-block:: python

    from zellij.core.variables import CatVar, ArrayVar, DynamicBlock
    from zellij.utils.neighborhoods import ArrayInterval, DynamicBlockInterval

    from evodags.search_algorithm.neighborhoods import LayersInterval, AdjMatrixHierarchicalInterval
    from evodags.search_space.dags import AdjMatrixVariable
    from evodags.search_space.variables import unitary_var, mlp_var, activation_var, create_int_var

    # We define the candidate operations for each nodes in the graph. Here we only consider multi-layers perceptron and identity operations.
    def operations_var(label, shape, size):
        return DynamicBlock(
            label,
            CatVar(
                label + "Candidates",
                [
                    unitary_var(label + " Unitary"),
                    mlp_var(label + " MLP"),
                ],
                neighbor=LayersInterval([2, 1]),
            ),
            size,
            neighbor=DynamicBlockInterval(neighborhood=2),
        )

    # We define the serach space, a graph handling one-dimensional data, and the final activation function before the prediction.
    def NN_monash_var(label="Neural Network", shape=1000, size=10):
        NeuralNetwork = ArrayVar(
            AdjMatrixVariable(
                "Cell",
                operations_var("Feed Cell", shape, size),
                neighbor=AdjMatrixHierarchicalInterval()
            ),
            activation_var("NN Activation"),
            create_int_var("Seed", None, 0, 10000),
            label=label,
            neighbor=ArrayInterval(),
        )
        return NeuralNetwork

    sp = NN_monash_var(m1_monthly_config["Lag"], size=3)
    

Once your search space is defined, you can draw random points:

.. code-block:: python
    p1,p2 = sp.random_point(), sp.random_point()
    print("First random point: ", p1)
    print("Second random point: ", p2)

See :ref:`sp` for more information.

Now we can use the loss function on the search space:

.. code-block:: python

  scores = loss([p1, p2])
  print(f"Best solution found:\nf({loss.best_point}) = {himmelblau.best_score}")
  print(f"Number of evaluations:{loss.calls}")
  print(f"All evaluated solutions:{loss.all_solutions}")
  print(f"All loss values:{loss.all_scores}")


Implementing an optimization strategy
=====================================

To ease the use of several metaheuristics, the user can directly use the function :func:`evodags.search_algorithm.pb_configuration.problem_configuration` to define its search strategy.
In our case we will use an Evolutionary Algorithm, we set the "MetaHeuristic" entry from the config to "GA".

.. code-block:: python
    
    evodags.search_algorithm.pb_configuration import problem_configuration
  
    exp_config = {
        "MetaHeuristic": "GA",
        "Generations": 2,
        "PopSize": 4,
        "MutationRate": 0.7,
        "TournamentRate": 10,
        "ElitismRate": 0.1,
        "RandomRate": 0.1,
    }

    _, search_algorithm = problem_configuration(exp_config, net, loss)

    best, score = search_algorithm.run()
    print(f"Best solution found:\nf({best}) = {score}")