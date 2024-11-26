.. _implemented_algorithms:


=============================
Search Algorithms implementation
=============================

`SearchAlgorithm` class
------------

The variables and their operators respectively defined in the `search space section <../Search_Space/index.rst>`_ and the `search operators section <../Search_Operators/index.rst>`_ sections can be used to implement various search algorithms.
**DRAGON** uses an abstract class called `SearchAlgorithm` to structure the algorithms.

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

This code is a highly simplified, schematic version of the `SearchAlgorithm` class to help illustrate its main aspects. 
The input arguments of the class depend on the application. 
The search space is a `variable` that can represent all considered configurations. 
The `evaluation` function takes a configuration as input, builds the model, trains and evaluates it, and then returns a `loss` value representing its performance. 
It's important to note that the class tries to minimize a value. 
The number `n_iterations` represents the amount of possible calls to the `evaluation` function during training.
The `SearchAlgorithm` class assumes that all search algorithms go through an initialization phase where a certain population is randomly generated and then evaluated. 
Then, until a maximum number of iterations is reached, a configuration is selected and evaluated by a `select_next_configuration` function specific to the algorithm in question. 
Depending on the performance obtained, the algorithm will process this configuration with the `process_evaluated_configuration` function.
A pseudo-code of a `select_next_configuration` function for the Evolutionary Algorithm is given below.

.. code-block:: python

   def select_next_configurations(self):
      parent_1, parent_2 = tournament_selection(self.population)
      offspring_1, offspring_2 = crossover(parent_1, parent_2)
      offspring_1 = self.search_space.neighbor(offspring_1)
      offspring_2 = self.search_space.neighbor(offspring_2)
      return [offspring_1, offspring_2]

It first selects two parents configurations using a tournament selection strategy.
The parents are then modified using a crossover and the mutations implemented as `neighbor` attributes.
When implementing a new search algorithm using the `SearchAlgorithm` structure, this function has to be specified.


Storage management for Deep Neural Networks
------------

The DAG coding objects such as the `Bricks`, `Nodes`, and `AdjMatrix` are all `nn.Module` that contain trained weights. 
Therefore, their size can take up a lot of memory. 
To prevent the system from handling too many neural networks at the same time, the configurations are cached while the search algorithms are running and only loaded for evaluation or to create a new configuration by mutating them. 
For this purpose, each configuration is assigned a unique number called `idx`. 
A `storage` dictionary summarizes all the information needed by the search algorithm for each configuration. 
All need `loss`, but when dealing with resource allocation, some algorithms may need additional information such as the number of resources a configuration has already received.
The `storage` dictionary is updated during the `process_evaluated_configuration`.
A pseudo-code of the one use by Mutant-UCB is given below.

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
The file-based backup system also makes it easy to resume an aborted optimisation. 
The `SearchAlgorithm' class incrementally saves a `.csv' file containing information about previously evaluated configurations. 
This file can be used to find the stage at which the optimisation was performed and to extend it. 
This is done using the `recover_optimisation` method of the class. 

Distributed version through MPI
------------

The class `SearchAlgorithm` allows to distribute the algorithms on multiple computation nodes from a High Performance Computing (HPC) achitectures.
This is based on a Message Passing Interface (MPI) thanks to the `mpi4py <https://mpi4py.readthedocs.io/en/stable/>`_ package.
The search algorithm relies on a master process and several workers processes, each assigned to a GPU.
The master process performs the algorithm main steps such as creating the population or selecting the next configurations.
The worker processes perform the training and evaluation of a configuration sent by the master.
An illustration of the implementation can be found below.

.. tikz::

   \tikzset{every picture/.style={line width=0.75pt}} %set default line width to 0.75pt        

   \begin{tikzpicture}[x=0.75pt,y=0.75pt,yscale=-1,xscale=1]
   %uncomment if require: \path (0,300); %set diagram left start at 0, and has height of 300


   % Text Node
   \draw (21,76.5) node [anchor=north west][inner sep=0.75pt]   [align=left] {\textbf{Master}\\→ Randomly draws configurations\\→ Selects the next configuration to evaluate\\	(may perform mutations, crossover...)\\→ Process the evaluated configuration\\→ Identify the best model};
   % Text Node
   \draw (406.96,3.08) node [anchor=north west][inner sep=0.75pt]  [color={rgb, 255:red, 144; green, 19; blue, 254 }  ,opacity=1 ,rotate=-359.96] [align=left] {\textbf{Worker 1 }\\→ Associated with GPU 1\\→ Evaluate the configuration sent \\by the master\\→ Store the configuration};
   % Text Node
   \draw (407,171) node [anchor=north west][inner sep=0.75pt]  [color={rgb, 255:red, 65; green, 117; blue, 5 }  ,opacity=1 ] [align=left] {\textbf{Worker \textit{n }}\\→ Associated with GPU \textit{n}\\→ Evaluate the configuration sent \\by the master\\→ Store the configuration};
   % Text Node
   \draw (295.04,55.69) node [anchor=north west][inner sep=0.75pt]  [rotate=-0.26] [align=left] {{\small Configuration}};
   % Text Node
   \draw (364.04,101.69) node [anchor=north west][inner sep=0.75pt]  [color={rgb, 255:red, 144; green, 19; blue, 254 }  ,opacity=1 ,rotate=-0.26] [align=left] {{\small Loss}};
   % Text Node
   \draw (332.04,199.69) node [anchor=north west][inner sep=0.75pt]  [color={rgb, 255:red, 144; green, 19; blue, 254 }  ,opacity=1 ,rotate=-0.26] [align=left] {{\small \textcolor[rgb]{0.25,0.46,0.02}{Loss}}};
   % Text Node
   \draw (336.04,149.69) node [anchor=north west][inner sep=0.75pt]  [rotate=-0.26] [align=left] {{\small Configuration}};
   % Connection
   \draw    (320,96.48) -- (402.02,76.99) ;
   \draw [shift={(403.96,76.53)}, rotate = 166.63] [color={rgb, 255:red, 0; green, 0; blue, 0 }  ][line width=0.75]    (10.93,-3.29) .. controls (6.95,-1.4) and (3.31,-0.3) .. (0,0) .. controls (3.31,0.3) and (6.95,1.4) .. (10.93,3.29)   ;
   % Connection
   \draw [color={rgb, 255:red, 144; green, 19; blue, 254 }  ,draw opacity=1 ] [dash pattern={on 0.84pt off 2.51pt}]  (403.96,86.81) -- (321.95,106.3) ;
   \draw [shift={(320,106.76)}, rotate = 346.63] [color={rgb, 255:red, 144; green, 19; blue, 254 }  ,draw opacity=1 ][line width=0.75]    (10.93,-3.29) .. controls (6.95,-1.4) and (3.31,-0.3) .. (0,0) .. controls (3.31,0.3) and (6.95,1.4) .. (10.93,3.29)   ;
   % Connection
   \draw    (320,168.24) -- (402.05,187.74) ;
   \draw [shift={(404,188.2)}, rotate = 193.37] [color={rgb, 255:red, 0; green, 0; blue, 0 }  ][line width=0.75]    (10.93,-3.29) .. controls (6.95,-1.4) and (3.31,-0.3) .. (0,0) .. controls (3.31,0.3) and (6.95,1.4) .. (10.93,3.29)   ;
   % Connection
   \draw [color={rgb, 255:red, 65; green, 117; blue, 5 }  ,draw opacity=1 ] [dash pattern={on 0.84pt off 2.51pt}]  (404,198.48) -- (321.95,178.98) ;
   \draw [shift={(320,178.52)}, rotate = 13.37] [color={rgb, 255:red, 65; green, 117; blue, 5 }  ,draw opacity=1 ][line width=0.75]    (10.93,-3.29) .. controls (6.95,-1.4) and (3.31,-0.3) .. (0,0) .. controls (3.31,0.3) and (6.95,1.4) .. (10.93,3.29)   ;

   \end{tikzpicture}

As the training and the evaluation of a neural network is the most time-consuming part of the search algorithm, this makes **DRAGON** easily scalable on HPC infrastructures. 
The `SearchAlgorithm` class activates by itself the `MPI` version by looking if the package `mpi4py` is available.

Implemented Algorithms
------------

While it is possible to implement new search algorithm by following or not the `SearchAlgorithm` strucuture, some are already implemented within the package and ready to be used.
A Random Search, HyperBand, an Evolutionary Algorithm and Mutant-UCB are indeed available.
For a given application, they require the implementation of a search space, an performance evaluation function and to set some required parameters such as the directory to save the configurations or the number of iterations.

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