.. _implemented_algorithms:


=============================
Presentation
=============================

Main structure
--------------

The variables and their operators can be used to implement various search algorithms.
DRAGON uses an abstract class called `SearchAlgorithm` to structure the algorithms.

.. code-block:: python
   
   class SearchAlgorithm:
      def __init__(self, search_space, n_iterations, population_size, evaluation):
        self.search_space = search_space
        self.n_iterations = n_iterations
        self.population_size = population_size
        self.evaluation = evaluation
        self.min_loss = np.inf

      def run(self):
         # Generate the first population of configurations
         self.population = self.create_first_population()

         # Evaluate and process each of these configurations
         for i,p in enumerate(self.population):
            loss = self.evaluation(idx)
            self.process_evaluated_configuration(idx, loss)
            if loss < self.min_loss:
               self.min_loss = loss

         t = population_size
         # While the number of iterations has not been reached
         while t < n_iterations:
            # Select the next configurations
            idx_list = self.select_next_configurations()
            for idx in idx_list:
               # Evaluate the configurations
               loss = self.evaluation(idx)
               # Process the evaluated configuration
               self.process_evaluated_configuration(idx, loss)
               if loss < self.min_loss:
                  self.min_loss = loss
               t+=1

This pseudo-code is a highly simplified, schematic version of the `SearchAlgorithm` class to help illustrate its main aspects.
The class input arguments depend on the application.
The search space is a (composed) `Variable` that can represent all considered configurations.
The `evaluation` function takes a configuration as input, builds the model, trains and evaluates it, and then returns a `loss` value representing its performance. This function depends on the tasks at hand and should be implemented by the user.
It is important to note that the class `SearchAlgorithm` is made to minimize a value, and thus the `evaluation` function should return a loss, not a reward.
The number `n\_iterations` represents the amount of calls to the `evaluation` function during training.
The `SearchAlgorithm` class assumes that all search algorithms go through an initialization phase where a certain population is randomly generated and evaluated.
Then, until a maximum number of iterations is reached, a configuration is selected and evaluated by a `select\_next\_configuration` function, specific to the algorithm in question.
Depending on the performance obtained, the algorithm will process this configuration with the `process\_evaluated\_configuration` function.
A pseudo-code of a `select\_next\_configuration` function for the Evolutionary Algorithm is given below.

.. code-block:: python

   def select_next_configurations(self):
      parent_1, parent_2 = tournament_selection(self.population)
      offspring_1, offspring_2 = crossover(parent_1, parent_2)
      offspring_1 = self.search_space.neighbor(offspring_1)
      offspring_2 = self.search_space.neighbor(offspring_2)
      return [offspring_1, offspring_2]

It first selects two parent configurations using a tournament selection strategy.
The parents are then modified using a crossover and the mutations implemented as `neighbor` attributes.
When implementing a new search algorithm using the `SearchAlgorithm` structure, this function has to be specified.


Storage management for Deep Neural Networks
--------------------------------------------

The objects encoding DAG such as the `Bricks`, `Nodes`, and `AdjMatrix` are all `nn.Module` that contain trained weights.
Therefore, they can take up a lot of memory.
To prevent the system from handling too many neural networks, the configurations are cached while the search algorithm is running and only loaded for evaluation or to create a new configuration by mutating them.
For this purpose, each configuration is assigned a unique number called `idx`.
A `storage` dictionary summarizes all the information needed by the search algorithm for each configuration.
All algorithms require the `loss` for each configuration. However, some algorithms leveraging resource allocation may need additional information such as the number of resources a configuration has already received.
The `storage` dictionary is updated during the `process\_evaluated\_configuration`.
A pseudo-code of the one used by Mutant-UCB is given below.

.. code-block::python
   
   def process_evaluated_configuration(self, idx, loss):
      if idx in self.storage.keys(): # If the configuration has already been implemented
         self.storage[idx]['Loss'] = loss
         self.storage[idx]['UCBLoss'] = (loss + self.storage[idx]['N_bar']*self.storage[idx]['UCBLoss'])/(self.storage[idx]['N_bar']+1)
         self.storage[idx]['N'] +=1 # Number of time the configuration has been picked
         self.storage[idx]['N_bar'] +=1 # Number of time the configuration has been trained
        else:
            self.storage[idx] = {"N": 1, "N_bar": 1, "UCBLoss": loss, "Loss": loss}

This function should also be specified when implementing a new search algorithm.
The file-based backup system also makes it easy to resume an aborted optimization.
The `SearchAlgorithm` class incrementally saves a CSV file containing information about previously evaluated configurations.
This file is used to find the stage where the search algorithm stopped and to continue the search. 
This is done using the `recover\_optimization` method of the class.


Distributed version through MPI
-------------------------------

The class `SearchAlgorithm`` allows the distribution of the algorithms on multiple computation nodes of a High Performance Computing (HPC) architecture.
This is based on a Message Passing Interface (MPI) with the `mpi4py package <https://mpi4py.readthedocs.io/en/stable/>`_.
The search algorithm relies on a master process and several workers processes, each assigned to a GPU.
The master process performs the algorithm's main steps such as creating the population or selecting the next configurations.
It dynamically sends configurations to the workers to evaluate. 
As soon as a process finishes an evaluation, the master processes the returned model and selects the next one to send to the worker.
The worker processes can be associated with a unique GPU. They perform the training and evaluation of the configuration sent by the master.
An illustration of the implementation can be found below.

.. tikz::

   \begin{tikzpicture}[
        every node/.style={align=left, font=\small},
        box/.style={draw, rounded corners, text width=5cm, minimum height=1.2cm, inner sep=5pt},
        dashed-arrow/.style={->, dashed, thick},
        arrow/.style={->, thick}
    ]

    % Nodes
    \node[box] (master) {\textbf{Master}\\
        → Randomly draws configurations\\
        → Selects the next configuration to evaluate (may perform mutations, crossover...)\\
        → Processes the evaluated configuration\\
        → Identifies the best model};

    \node[box, above right=1cm and 2cm of master.east, text=output_red] (worker1) {\textbf{Worker 1}\\
        → Associated with GPU 1\\
        → Evaluates the configuration sent by the master\\
        → Stores the configuration};

    \node[box, below right=1cm and 2cm of master.east, text=ulcolour] (workern) {\textbf{Worker \textit{n}}\\
        → Associated with GPU \textit{n}\\
        → Evaluates the configuration sent by the master\\
        → Stores the configuration};

    % Arrows
    \draw[->, thick, bend left=15] (master.east) to node[above, yshift=9mm] {Configuration} (worker1.west);
    \draw[->, thick, dashed, bend left=15, output_red] (worker1.west) to node[below right] {Loss} (master.east);

    \draw[->, thick, bend left=15] (master.east) to node[midway, above, xshift=6mm, yshift=2mm] {Configuration} (workern.west);
    \draw[->, dashed, thick, bend left=15, ulcolour] (workern.west) to node[midway, below, yshift=-3mm, xshift=-1mm] {Loss} (master.east);

    \end{tikzpicture}

As training a neural network is the most time-consuming part of the search algorithm, distributing the part to several devices makes DRAGON search algorithms more efficient and easily scalable on HPC infrastructures.
The `SearchAlgorithm` class activates by itself the MPI version by looking if the package mpi4py is available.

Implemented Algorithms
-----------------------

It is possible to implement new search algorithms that extend or not the `SearchAlgorithm` class. 
However, some are already available within the package and ready to be used.
A Random Search, HyperBand, an Evolutionary Algorithm, and, Mutant-UCB are implemented.
For a given application they require the implementation of a search space, a performance evaluation function and the setting of some required parameters such as the path to save the configurations, or the number of iterations.

.. code-block:: python
   from dragon.search_algorithm.mutant_ucb import Mutant_UCB

   search_algorithm = Mutant_UCB(search_space, save_dir="save/test_mutant", T=20, N=5, K=5, E=0.01, evaluation=loss_function)
   search_algorithm.run()

For the detailed implementations of these algorithms, see:

.. toctree::
   :maxdepth: 1 

   random_search
   ssea
   mutant_ucb
   hyperband
   
..