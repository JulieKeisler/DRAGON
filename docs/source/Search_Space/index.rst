.. _search_space:

=============================
Presentation
=============================

The search space design is based on an abstract class called `Variable`, originally proposed within a hyperparameters optimization package called `zellij <https://zellij.readthedocs.io/en/latest/>`_. 
This class can represent various objects from float to tree-based or array-like structures.
Each child class should implement a `random`` method, which represents the rules defining the variable and how to generate random values. 
The other method to implement is called `isconstant`` and specifies whether the variable is a constant or not.
Variables can be composed to represent more or less complex objects.
Among the composed variables, some have been created specifically for the DAG encodings.
A variable can be extended by `Addons` to implement additional features such as the search operators detailed `here <../Search_Operators/index.rst>`.
The structure of a variable definition is the following:

.. code-block:: python

   from dragon.search_space.base_variables import Variable

   class CustomVar(Variable):

      def __init__(self, label, **kwargs):
         super(CustomVar, self).__init__(label, **kwargs)
         """
         The label is the unique idenfier of a variable within the search space.
         Initialize the variable attribute.
         """

    def random(self, size=None):
        """
        Create `size` random values of the variables
        """

    def isconstant(self):
        """
        Specify is a variable is a constant or not. 
        This function might depends on the variable attributes.
        """

All the variables ihnerit from the abstract class `Variable`.
An example of the implementation of a variable for an integer can be found below:

.. code-block:: python

   class IntVar(Variable):
      """
      `IntVar` defines a variable discribing Integer variables. 
      The user must specify a lower and an upper bounds.
      """
      def __init__(
         self, label, lower, upper, **kwargs):
         super(IntVar, self).__init__(label, **kwargs)
         self.low_bound = lower
         self.up_bound = upper + 1

      def random(self, size=None):
         """
         `size` integers are randomly drawn form the interval `[low_bound, up_bound]`.
         """
         return np.random.randint(self.low_bound, self.up_bound, size, dtype=int)

      def isconstant(self):
         """
         An IntVar is a constant if the upper and the lower bounds are equals.
         """
         return self.up_bound == self.low_bound

This class can be used to create integers:

.. code-block:: python

   v = IntVar("An integer variable", 0, 5)
   v.random()
   3

In this example, the variable `v`` defines an integer taking values from 0 to 5. 
When calling `v.random()`, the script returns an integer from this range, here 3.

**DRAGON** offers the implementage of base and composed variables to create more or less complex search spaces.
Among the composed variables, some have been created specifically for the DAG-encodings.

Base variables
------------

Deep Neural Networks are made of layers. 
In **DRAGON**’s case, those layers are `nn.Module` from *PyTorch*.
The user can integrate any base or custom `nn.Module`, but has to wrap it into a `Brick`` object.
This *Python* class takes an input shape and some hyperparameters as arguments and initializes a given `nn.Module` with these hyperparameters so it can process a tensor of the specified input shape.
The forward pass of a `Brick` can directly apply the layer to an input tensor, or it can be more complex and transform the input data before the operation.
Finally, the abstract class `Brick` also implements a `modify\_operation` method.
It takes an `input\_shape` and modifies the shape of the operation weights so that it can take as input a tensor of shape `input\_shape`.
This method is applied when the Deep Neural Network is created or modified.
The use cases will be detailed below.

.. list-table:: Base variables
   :widths: 25 25 50
   :header-rows: 1

   * - Type
     - Variable Name
     - Main parameters
   * - Integer
     - `IntVar`
     - Lower / upper bound
   * - Float
     - `FloatVar`
     - Lower / upper bound
   * - Categorical (string, etc)
     - `CatVar`
     - Features: list of possible choices
   * - Constant (any object)
     - `Constant`
     - Value: constant value

Note that the features from a `CatVar` variable might include `Variables` and non-variables values. 
The implementation of the base variables is detailed in the `Base Variables <base_variables.rst>`_ section.

.. toctree::
   :maxdepth: 1

   base_variables

Composed variables
------------

The base variables can be composed to create more complex objects such as arrays of variables.

.. code-block:: python

   from dragon.search_space.base_variables import ArrayVar, IntVar, FloatVar, CatVar

   a = ArrayVar(IntVar("int_1", 0,8),
    ...              IntVar("int_2", 4,45),
    ...              FloatVar("float_1", 2,12),
    ...              CatVar("cat_1", ["Hello", 87, 2.56]))
   a.random()
   [5, 15, 8.483221226216427, 'Hello']

Here we have created an array of four different elements: two integers, one between 0 and 8 and the other between 4 and 45, a float between 2 and 12, and a categorical variable that takes values within `[{"Hello"}, 87, 2.56]`.
For example, a `ArrayVar` can represent a list of hyperparameters of a machine learning model.
Unlike the `CatVar` features attribute that may contain `Variable` and non- `Variable` elements, the attributes of composed variables must inherit from the class `Variable`.
This means that we cannot create an `ArrayVar` with a simple 5 as argument.
To include a constant integer, we have to encode it using a `Constant` variable.

.. list-table:: Composed variables
   :widths: 25 25 50
   :header-rows: 1

   * - Definition
     - Variable Name
     - Main parameters
   * - Array of Variables
     - `ArrayVar`
     - List of `Variables`
   * - Fix number of repeats
     - `Block`
     - `Variable` that will be repeated and the number of repetitions.
   * - Random number of repeats
     - `DynamicBlock`
     - `Variable` that will be repeated and the maximum number of repetitions.

The implementation of the composed variables is detailed in the `Composed Variables <composed_variables.rst>`_ section.

.. toctree::
   :maxdepth: 1

   composed_variables

Deep Neural Networks Encoding
------------

Both the base and composed variables have been used to encode Deep Neural Networks architecture and hyperparameters.

Operations and hyperparameters encoding
~~~~~~~~~~~~~~~~~~~~~~

The Deep Neural Networks are made of layers. In **DRAGON**'s case, those layers are *nn.Module* from *PyTorch*.
The user can use any base or custom *nn.Module*, but as to wrap it into a *Brick* object. 
A brick takes as input an input shape and some hyperparameters and initialize a given *nn.Module* with these hyperparameters so it can pocess a tensor of the given input shape.
The forward pass of a *Brick* can just apply the layer to an input tensor, or be more complex to transform the input data before the operation.
Finally, the abstract class *Brick* also implements a *modify_operation* method. 
It takes as input an `input_shape` tuple and modifies the operation weights shape, so that the operation may take as input a vector of shape `input_shape`.
This method is applied when the Deep Neural Network is created or modified.
The applications case will be detailled below.

+--------------------------------------------------+--------------------------------------------------------------+
|                                                  |                                                              |
|.. code-block:: python                            |.. code-block:: python                                        |
|                                                  |                                                              |
|  import torch.nn as nn                           | from dragon.search_space.cells import Brick                  |
|  from dragon.search_space.cells import Brick     | import torch.nn as nn                                        |
|                                                  |                                                              |
|  class Dropout(Brick):                           | class MLP(Brick):                                            |
|     def __init__(self, input_shape, rate):       |    def __init__(self, input_shape, out_channels):            |
|        super(Dropout, self).__init__(input_shape)|       super(MLP, self).__init__(input_shape)                 |
|        self.dropout = nn.Dropout(p=rate)         |       self.in_channels = input_shape[-1]                     |
|     def forward(self, X):                        |       self.linear = nn.Linear(self.in_channels, out_channels)|
|        X = self.dropout(X)                       |    def forward(self, X):                                     |
|        return X                                  |       X = self.linear(X)                                     |
|     def modify_operation(self, input_shape):     |       return X                                               |
|        pass                                      |    def modify_operation(self, input_shape):                  |
|                                                  |       d_in = input_shape[-1]                                 |
|                                                  |       diff = d_in - self.in_channels                         |
|                                                  |       sign = diff / np.abs(diff) if diff !=0 else 1          |
|                                                  |       pad = (int(sign * np.ceil(np.abs(diff)/2)),            |
|                                                  |              int(sign * np.floor(np.abs(diff))/2))           |
|                                                  |       self.in_channels = d_in                                |
|                                                  |       self.linear.weight.data =                              |
|                                                  |             nn.functional.pad(self.linear.weight, pad)       |
+--------------------------------------------------+--------------------------------------------------------------+

The two blocks of code above show the implementation of a `Dropout` layer and an `MLP` (Multi-Layer Perceptron), respectively.
While the wrapping of the `Dropout` layer into a `Brick` object requires minimal modifications, the `MLP` wrapping necessitates some effort to implement the `modify\_operation` method.
Indeed, the shape of the weights of an `nn.Linear` operation depends on the input tensor dimension.

The variable encoding a `Brick` is called `HpVar`.
It takes as input a `Constant` or a `CatVar` containing a single or several layers implemented a `Bricks`, as well as a dictionary of hyperparameters.
If a `CatVar` is given as input operation, all the `Bricks` contained in the `CatVar`'s features should share the same hyperparameters.

+--------------------------------------------------------------------+-----------------------------------------------------------------------+
|.. code-block:: python                                              |.. code-block:: python                                                 |
|                                                                    |                                                                       |
|  from dragon.search_space.bricks import MLP                        |  from dragon.search_space.bricks import LayerNorm1d, BatchNorm1d      |
|  from dragon.search_space.base_variables import Constant, IntVar |  from dragon.search_space.base_variables import CatVar              |
|  from dragon.search_space.dragon_variables import HpVar            |  from dragon.search_space.dragon_variables import HpVar               |
|                                                                    |                                                                       |
|  mlp = Constant("MLP operation", MLP)                              |  norm = CatVar("1d norm layers", features=[LayerNorm1d, BatchNorm1d]) |
|  hp = {"out_channels": IntVar("out_channels", 1, 10)}              |  norm_var = HpVar("Norm var", norm, hyperparameters={})               |
|  mlp_var = HpVar("MLP var", mlp, hyperparameters=hp)               |  norm_var.random()                                                    |
|  mlp_var.random()                                                  |  [<class 'dragon.search_space.bricks.normalization.BatchNorm1d'>, {}] |
|  [<class 'dragon.search_space.bricks.basics.MLP'>,                 |                                                                       |
|     {'out_channels': 9}]                                           |                                                                       |
+--------------------------------------------------------------------+-----------------------------------------------------------------------+

TThese two examples show how to use `HpVar` with a `Constant` and a `CatVar` operation respectively.
Here, the `CatVar` is made here of two versions of normalization layers which share the same hyperparameters (none in this example).
The only hyperparameter that can be optimized for the `MLP` layer is the size of the output channel, here, an integer between $1$ and $10$.
To facilitate the use of DRAGON, operations (such as convolution, attention, pooling, or identity layers) as `Brick` and their variable `HpVar` are already implemented in the package.
To facilitate the use of **DRAGON**, operations as `Brick` and their variable `HpVar` are already implemented in the package and detailed in the `bricks section <bricks.rst>`_.

.. toctree::
   :maxdepth: 1

   bricks

Node encoding
~~~~~~~~~~~~~~~~~~~~~~

**DRAGON** implements Deep Neural Networks as computational graphs, where each node is a succession of a combiner, an operation and an activation function, as detailed in \cref{chap:dragon_jmlr}.
The operation is encoded as a `Brick`, as mentioned above.
The combiner unifies the (potential) multiple inputs the node might have into one unique tensor.
The combiners available in DRAGON are *add*, *mul* and *concat* and are encoded as strings.
The activation function can be any *PyTorch* implemented or custom activation function.
An `nn.Module` object called `Node` takes as inputs these three elements to create a node.
A `Node` implements several methods. The main ones are:
* `set_operation`: takes as input a variable `input_shapes` containing the input shapes of the incoming tensors. 
The method use the combiner to compute the operation input shape and initialize the operation weights with the right shape. 
The initialized operation is then used to compute the node output shape.  
This value will be used by the `set_operations` methods from the child nodes of the current one.
* `modification`: modify the node combiner, operation, hyperparameters or output shape. 
The modification may happened after a mutation or because the tensor input shape has changed. 
If the operation is not modified, the method `modify_operation` from the `Brick` operation is called to only change the weights. 
* `set`: automatically choose between the `set_operation` and `modification` methods.
* `forward`: compute the node forward pass from the combiner to the activation function.

The `Variable` corresponding to a `Node` is called `NodeVariable`.
It takes as input a `Constant` or a `CatVar` for the combiner and the activation functions.
The operation is implemented as an `HpVar` as mentioned above.
However, a node can have multiple candidate operations, all of them implemented as different `HpVar` objects.
In this case, instead of directly being given as an `HpVar`, they are contained within a `CatVar`.
The `CatVar` features will contain the different `HpVar`.
An example is given below.

.. code-block:: python

   from dragon.search_space.base_variables import CatVar
   from dragon.search_space.bricks_variables import activation_var
   operation=CatVar("Candidates", [mlp_var, norm_var])
   candidates = NodeVariable(label = "Candidates", 
               combiner=CatVar("Combiner", features=['add', 'concat']),
               operation=CatVar("Candidates", [mlp, pooling]),
               activation_function=activation_var("Activation"))

The activation functions are encoded through the `activation\_var` object.
It is a default `CatVar` implemented within DRAGON which contains the basic activation functions from *PyTorch*.
The `random` method from the `NodeVariable` randomly selects a combiner, an activation function, an operation (in case of a `CatVar` operation), and draws random hyperparameters.

DAG encoding
~~~~~~~~~~~~~~~~~~~~~~

Finally, the last structure presented is the DAG, which (partially) encodes a Deep Neural Network.
The object that encodes the graphs is called `AdjMatrix` and is also a `nn.Module`. It takes as arguments a list of nodes and an adjacency matrix (a two-dimensional array) representing the edges between these nodes.
A method `assert\_adj\_matrix` is used to evaluate the correct format of the adjacency matrix (e.g, right number of rows and columns, upper triangular, diagonal full of zeros).
The directed acyclic structure of the graph allows ordering the nodes as explained in \cref{chap:dragon_jmlr}.
Just like the `Node` object, the `AdjMatrix` implements a method `set` that takes as an argument `input\_shape` and calls the method `set` from each node in that order.
The `forward` pass computation is also done in this order.
During the forward computation, the outputs are stored in a list to be used for later nodes of the graph that have them as input.

The `Variable` that represents the `AdjMatrix` is called `EvoDagVariable`.
It takes as input a `DynamicBlock` whose repeated variable would be a `NodeVariable`.
This `NodeVariable` will have its operation encoded as `CatVar` in case of multiple candidate layers.
A random `AdjMatrix` is created by first drawing the number of nodes from the graph. Then a random value of `NodeVariable` is drawn for each node.
Finally an `AdjMatrix` of the right dimension is created.



Implementation
~~~~~~~~~~~~~~~~~~~~~~

The implementation of the objects and variables use to encode the Deep Neural Networks (e.g `Brick`, `Node`, `EvoDagsVariable`) is detailed in `This section <dag_encoding.rst>`_.

.. toctree::
   :maxdepth: 1

   dag_en

The Figure below illustrates how the elements are linked together. 
The hierarchical composition of the variables creating a DAG allows optimization at various levels, from the graph structure to any operation hyperparameters. 
Hyperparameters or operations can be imposed to reduce the search space by passing certain operations with constant hyperparameters `Constant`. 
This way, we can reconstruct more constrained search spaces close to cell-based search spaces.

.. tikz::

   \tikzset{every picture/.style={line width=0.75pt` %set default line width to 0.75pt        

   \begin{tikzpicture}[x=0.75pt,y=0.75pt,yscale=-1,xscale=1]
   %uncomment if require: \path (0,375); %set diagram left start at 0, and has height of 375


   % Text Node
   \draw (14.75,44) node [anchor=north west][inner sep=0.75pt]   [align=left] {\begin{minipage}[lt]{80.81pt}\setlength\topsep{0pt}
   \begin{center}
   \textcolor[rgb]{0.29,0.56,0.89}{AdjMatrix}\\=\\\textcolor[rgb]{0.56,0.07,1}{EvoDagsVariable}
   \end{center}

   \end{minipage`;
   % Text Node
   \draw (229,4) node [anchor=north west][inner sep=0.75pt]   [align=left] {Matrix: adjacency matrix representing \\the edges between the node};
   % Text Node
   \draw (229,64) node [anchor=north west][inner sep=0.75pt]   [align=left] {Operations: list of \textcolor[rgb]{0.29,0.56,0.89}{Nodes`;
   % Text Node
   \draw (222,87.5) node [anchor=north west][inner sep=0.75pt]   [align=left] {\begin{minipage}[lt]{67.35pt}\setlength\topsep{0pt}
   \begin{center}
   =\\\textcolor[rgb]{0.56,0.07,1}{DynamicBlock}
   \end{center}

   \end{minipage`;
   % Text Node
   \draw (329,87.5) node [anchor=north west][inner sep=0.75pt]   [align=left] {\begin{minipage}[lt]{63.81pt}\setlength\topsep{0pt}
   \begin{center}
   =\\\textcolor[rgb]{0.56,0.07,1}{NodeVariable}
   \end{center}

   \end{minipage`;
   % Text Node
   \draw (27.25,163) node [anchor=north west][inner sep=0.75pt]   [align=left] {\begin{minipage}[lt]{63.81pt}\setlength\topsep{0pt}
   \begin{center}
   \textcolor[rgb]{0.29,0.56,0.89}{Node}\\=\\\textcolor[rgb]{0.56,0.07,1}{NodeVariable}
   \end{center}

   \end{minipage`;
   % Text Node
   \draw (229,144) node [anchor=north west][inner sep=0.75pt]   [align=left] {Combiner = \ \textcolor[rgb]{0.56,0.07,1}{Constant} or \textcolor[rgb]{0.56,0.07,1}{CatVar`;
   % Text Node
   \draw (229,184.5) node [anchor=north west][inner sep=0.75pt]   [align=left] {Operation and hyperparameters = \ \textcolor[rgb]{0.56,0.07,1}{HpVar`;
   % Text Node
   \draw (229,225) node [anchor=north west][inner sep=0.75pt]   [align=left] {Activation function = \textcolor[rgb]{0.56,0.07,1}{Constant} or \textcolor[rgb]{0.56,0.07,1}{CatVar`;
   % Text Node
   \draw (14.25,282) node [anchor=north west][inner sep=0.75pt]   [align=left] {\begin{minipage}[lt]{81.6pt}\setlength\topsep{0pt}
   \begin{center}
   Operation and hyperparameters\\=\\\textcolor[rgb]{0.56,0.07,1}{HpVar}
   \end{center}

   \end{minipage`;
   % Text Node
   \draw (229,282) node [anchor=north west][inner sep=0.75pt]   [align=left] {\textcolor[rgb]{0.29,0.56,0.89}{Brick} or list of \textcolor[rgb]{0.29,0.56,0.89}{Bricks }(*PyTorch} operation) = \textcolor[rgb]{0.56,0.07,1}{Constant} or \textcolor[rgb]{0.56,0.07,1}{CatVar`;
   % Text Node
   \draw (229,324) node [anchor=north west][inner sep=0.75pt]   [align=left] {Hyperparameters = dictionnary of base variables \\(e.g: \textcolor[rgb]{0.56,0.07,1}{FloatVar}, \textcolor[rgb]{0.56,0.07,1}{CatVar})};
   % Connection
   \draw    (134.75,62.55) -- (225.8,46.35) ;
   \draw [shift={(227.77,46)}, rotate = 169.91] [color={rgb, 255:red, 0; green, 0; blue, 0 }  ][line width=0.75]    (10.93,-3.29) .. controls (6.95,-1.4) and (3.31,-0.3) .. (0,0) .. controls (3.31,0.3) and (6.95,1.4) .. (10.93,3.29)   ;
   % Connection
   \draw    (134.75,73.24) -- (224,72.87) ;
   \draw [shift={(226,72.86)}, rotate = 179.76] [color={rgb, 255:red, 0; green, 0; blue, 0 }  ][line width=0.75]    (10.93,-3.29) .. controls (6.95,-1.4) and (3.31,-0.3) .. (0,0) .. controls (3.31,0.3) and (6.95,1.4) .. (10.93,3.29)   ;
   % Connection
   \draw    (122.25,192.58) -- (224,192.76) ;
   \draw [shift={(226,192.76)}, rotate = 180.1] [color={rgb, 255:red, 0; green, 0; blue, 0 }  ][line width=0.75]    (10.93,-3.29) .. controls (6.95,-1.4) and (3.31,-0.3) .. (0,0) .. controls (3.31,0.3) and (6.95,1.4) .. (10.93,3.29)   ;
   % Connection
   \draw    (122.25,185.07) -- (252.6,165.3) ;
   \draw [shift={(254.58,165)}, rotate = 171.38] [color={rgb, 255:red, 0; green, 0; blue, 0 }  ][line width=0.75]    (10.93,-3.29) .. controls (6.95,-1.4) and (3.31,-0.3) .. (0,0) .. controls (3.31,0.3) and (6.95,1.4) .. (10.93,3.29)   ;
   % Connection
   \draw    (122.25,199.43) -- (272.68,220.72) ;
   \draw [shift={(274.66,221)}, rotate = 188.05] [color={rgb, 255:red, 0; green, 0; blue, 0 }  ][line width=0.75]    (10.93,-3.29) .. controls (6.95,-1.4) and (3.31,-0.3) .. (0,0) .. controls (3.31,0.3) and (6.95,1.4) .. (10.93,3.29)   ;

   \end{tikzpicture}