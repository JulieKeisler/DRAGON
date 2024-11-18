.. _search_space:

=============================
Presentation
=============================

The search space design is based on an abstract class called *Variable*, originally proposed within an hyperparameters optimization package called `zellij <https://zellij.readthedocs.io/en/latest/>`_.
A variable should implements a *random* method detailing how to create a random value and an *isconstant* method specifying if the variable is a constant or not.
A variable can take *Addons* to implement additional features such as the `Search Operators <../Search_Operators/index.rst>`_.
The search space is made of base and composed variables to create more or less complex search spaces.
Among the composed variables, some have been created specifically for the DAG-encodings.

Base variables
------------

The base variables implements basic objects such as integers, floats or categorical variables. Each of this object is associated with a *Variable*, which defines what values an object can take.
For example, an integer object will be associated with the *variable* `IntVar`, that will take as arguments the lower and upper bounds, defining where the integer is defined.

.. code-block:: python

   from dragon.search_space.zellij_variables import IntVar

   v = IntVar("An integer variable", 0, 5)
   v.random()
   
   3

In this example, the variable `v` defines an integer which can take values from 0 to 5. When calling `v.random()`, the script returns an integer from this interval, here `3`.
The Base variables available within **DRAGON** are listed below.

.. list-table:: Base variables
   :widths: 25 25 50
   :header-rows: 1

   * - Type
     - Variable Name
     - Main parameters
   * - Integer
     - `IntVar`
     - lower / upper bound
   * - Float
     - `FloatVar`
     - lower / upper bound
   * - Categorical (string, etc)
     - `CatVar`
     - features: list of possible choices
   * - Constant (any object)
     - `Constant`
     - Value: constant value

Note that the features from a `CatVar` variable might include `Variables` and non-variables values. 
The implementation of the base variables is detailed in the `Base Variables <base_variables.rst>`_ section.


Composed variables
------------

The base variables can be composed to create more complex objects such as arrays of variables.

.. code-block:: python

   from dragon.search_space.zellij_variables import ArrayVar, IntVar, FloatVar, CatVar

   a = ArrayVar(IntVar("int_1", 0,8),
    ...              IntVar("int_2", 4,45),
    ...              FloatVar("float_1", 2,12),
    ...              CatVar("cat_1", ["Hello", 87, 2.56]))
   a.random()
   [5, 15, 8.483221226216427, 'Hello']

Here we created an array made of four distinct elements: two integers respectively between 0 and 8 and 4 and 45, a float between 2 and 12 and a categorical variable which can take values within ["Hello", 87, 2.56].
An ArrayVar may represent a list of hyperparameters of a given machine learning for example.
In opposition to the `CatVar` features attribute which might mix variables and non-variables elements, the attributes from the composed variables have to be variables. 
It means we cannot create an array with a simple 5. 
To inclue a constant integer, we have to use the `Constant` variable.

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
|.. code-block::                                   |.. code-block::                                               |
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

The codes just above show respectively the implementation of a `Dropout` and an `MLP` layer. 
While the wrapping of the `Dropout` layer into a `Brick` object requires minimal modifications, the `MLP` wrapping necessitates some effort to implement the `modify_operation` layer.
Indeed, the weights of an `nn.Linear` shape layer depends on the input tensor dimension.

The variable encoding a `Brick` is called `HpVar`. 
It takes as input a `Constant` or a `CatVar` containing a single `Brick` or several `Bricks` representing the candidate operations, as well as a dictionary of hyperparameters..
If a `CatVar` is given as input operation, all the `Bricks` contained in the `CatVar` features should share the same hyperparameters.

+--------------------------------------------------------------------+-----------------------------------------------------------------------+
|.. code-block::                                                     |.. code-block::                                                        |
|                                                                    |                                                                       |
|  from dragon.search_space.bricks import MLP                        |  from dragon.search_space.bricks import LayerNorm1d, BatchNorm1d      |
|  from dragon.search_space.zellij_variables import Constant, IntVar |  from dragon.search_space.zellij_variables import CatVar              |
|  from dragon.search_space.dragon_variables import HpVar            |  from dragon.search_space.dragon_variables import HpVar               |
|                                                                    |                                                                       |
|  mlp = Constant("MLP operation", MLP)                              |  norm = CatVar("1d norm layers", features=[LayerNorm1d, BatchNorm1d]) |
|  hp = {"out_channels": IntVar("out_channels", 1, 10)}              |  norm_var = HpVar("Norm var", norm, hyperparameters={})               |
|  mlp_var = HpVar("MLP var", mlp, hyperparameters=hp)               |  norm_var.random()                                                    |
|  mlp_var.random()                                                  |  [<class 'dragon.search_space.bricks.normalization.BatchNorm1d'>, {}] |
|  [<class 'dragon.search_space.bricks.basics.MLP'>,                 |                                                                       |
|     {'out_channels': 9}]                                           |                                                                       |
+--------------------------------------------------------------------+-----------------------------------------------------------------------+

These two examples show how to use `HpVar` with a `Constant` and a `CatVar` operation respectively.
The `CatVar` is made here of two versions of normalization layers which share the same hyperparameters (none here).
The only hyperparameter that can be optimized for the `MLP` layer is the size of the output channel.
It is here an integer between 1 and 10.
To facilitate the use of **DRAGON**, operations as `Brick` and their variable `HpVar` are already implemented in the package and detailed in the `bricks section <bricks.rst>`_.

Node encoding
~~~~~~~~~~~~~~~~~~~~~~

**DRAGON** implements Deep Neural Networks as computational graphs, where the nodes are a succession of a combiner, an operation and an activation function.
The operation is implemented as a `Brick`, as mentionned above.
The combiner is used to unify the (possible) multiple inputs that the node might have into one unique tensor.
The combiners available in **DRAGON** are *add*, *mul* and *concat* and are encoded as a string.
The activation function can be any `PyTorch` implemented or custom activation function.
An `nn.Module` object called `Node` takes as input these three elements to create a node.
A `Node` implements a lot of methods. The main ones are:
- `set_operation`: takes as input a variable `input_shapes` containing the input shapes of the incoming tensors. The method use the combiner to compute the operation input shape and initialize the operation weights with the right shape. The initialized operation is then used to compute the node output shape.  This value will be used by the `set_operations` methods from the child nodes of the current one.
- `modification`: modify the node combiner, operation, hyperparameters or output shape. The modification may happened after a mutation or because the tensor input shape has changed. If the operation is not modified, the method `modify_operation` from the `Brick` operation is called to only change the weights. 
- `set`: automatically choose between the `set_operation` and `modification` methods.
- `forward`: compute the node forward pass from the combiner to the activation function.

The variable corresponding to a `Node` is called `NodeVariable.` 
It takes as input a `Variables` for the combiner and the activation functions which may be `Constant` or `CatVar`.
The operation is implemented an `HpVar` as mentioned above. 
In some cases, the node might take several candidate operations. Therefore, the operation is encoded as a `CatVar` of `HpVar`, containing the various candidates. 
The `random` method from the `NodeVariable` randomly select a combiner and an activation function. Then it randomly selects the operation (in case of a `CatVar` operation) and drawn random hyperparameters.

DAG encoding
~~~~~~~~~~~~~~~~~~~~~~

Finally, the last structure that has to be presented is the Directed Acyclic Graph which (partially) encodes a Deep Neural Network.
The object encoding the graphs is called `AdjMatrix` and is also a `nn.Module`. It takes as input a list of nodes and an adjacency matrix representing the edges between those nodes. 
A method `assert_adj_matrix` is used to assess the good format of the adjacency matrix (e.g, right number of rows and columns, upper-triangular, diagonal full of zeros).
The directed acyclic structure of the graph allow an ordering of the nodes. 
Just like the `Node` object, the `AdjMatrix` implements a method `set` which takes as input a argument `input_shape` and call the method `set` from each node following this order.
The `forward` pass computation is also made following this order.
During the forward computation, the outputs are stored in a list to be used for later nodes- from the graph having them as input.

The `Variable` able to create random `AdjMatrix` is called `EvoDagVariable`. 
It takes as input a `DynamicBlock` whose repeated variable would be a `NodeVariable`.
Usually this `NodeVariable` will have its operation encoded as a `CatVar` to have several candidate layers.
A random `AdjMatrix` is created by first randomly drawing the number of nodes from the graph. Then, a random value of the `NodeVariable` is drawn for each node.
Finally, an adjacency matrix of the right dimension is created.

The implementation of the objects and variables use to encode the Deep Neural Networks (e.g `Brick`, `Node`, `EvoDagsVariable`) is detailed in `This section <dag_encoding.rst>`_.
The Figure below illustrates how the elements are linked together.

.. tikz::

   \tikzset{every picture/.style={line width=0.75pt}} %set default line width to 0.75pt        

   \begin{tikzpicture}[x=0.75pt,y=0.75pt,yscale=-1,xscale=1]
   %uncomment if require: \path (0,375); %set diagram left start at 0, and has height of 375


   % Text Node
   \draw (14.75,44) node [anchor=north west][inner sep=0.75pt]   [align=left] {\begin{minipage}[lt]{80.81pt}\setlength\topsep{0pt}
   \begin{center}
   \textcolor[rgb]{0.29,0.56,0.89}{AdjMatrix}\\=\\\textcolor[rgb]{0.56,0.07,1}{EvoDagsVariable}
   \end{center}

   \end{minipage}};
   % Text Node
   \draw (229,4) node [anchor=north west][inner sep=0.75pt]   [align=left] {Matrix: adjacency matrix representing \\the edges between the node};
   % Text Node
   \draw (229,64) node [anchor=north west][inner sep=0.75pt]   [align=left] {Operations: list of \textcolor[rgb]{0.29,0.56,0.89}{Nodes}};
   % Text Node
   \draw (222,87.5) node [anchor=north west][inner sep=0.75pt]   [align=left] {\begin{minipage}[lt]{67.35pt}\setlength\topsep{0pt}
   \begin{center}
   =\\\textcolor[rgb]{0.56,0.07,1}{DynamicBlock}
   \end{center}

   \end{minipage}};
   % Text Node
   \draw (329,87.5) node [anchor=north west][inner sep=0.75pt]   [align=left] {\begin{minipage}[lt]{63.81pt}\setlength\topsep{0pt}
   \begin{center}
   =\\\textcolor[rgb]{0.56,0.07,1}{NodeVariable}
   \end{center}

   \end{minipage}};
   % Text Node
   \draw (27.25,163) node [anchor=north west][inner sep=0.75pt]   [align=left] {\begin{minipage}[lt]{63.81pt}\setlength\topsep{0pt}
   \begin{center}
   \textcolor[rgb]{0.29,0.56,0.89}{Node}\\=\\\textcolor[rgb]{0.56,0.07,1}{NodeVariable}
   \end{center}

   \end{minipage}};
   % Text Node
   \draw (229,144) node [anchor=north west][inner sep=0.75pt]   [align=left] {Combiner = \ \textcolor[rgb]{0.56,0.07,1}{Constant} or \textcolor[rgb]{0.56,0.07,1}{CatVar}};
   % Text Node
   \draw (229,184.5) node [anchor=north west][inner sep=0.75pt]   [align=left] {Operation and hyperparameters = \ \textcolor[rgb]{0.56,0.07,1}{HpVar}};
   % Text Node
   \draw (229,225) node [anchor=north west][inner sep=0.75pt]   [align=left] {Activation function = \textcolor[rgb]{0.56,0.07,1}{Constant} or \textcolor[rgb]{0.56,0.07,1}{CatVar}};
   % Text Node
   \draw (14.25,282) node [anchor=north west][inner sep=0.75pt]   [align=left] {\begin{minipage}[lt]{81.6pt}\setlength\topsep{0pt}
   \begin{center}
   Operation and hyperparameters\\=\\\textcolor[rgb]{0.56,0.07,1}{HpVar}
   \end{center}

   \end{minipage}};
   % Text Node
   \draw (229,282) node [anchor=north west][inner sep=0.75pt]   [align=left] {\textcolor[rgb]{0.29,0.56,0.89}{Brick} or list of \textcolor[rgb]{0.29,0.56,0.89}{Bricks }(\textit{PyTorch} operation) = \textcolor[rgb]{0.56,0.07,1}{Constant} or \textcolor[rgb]{0.56,0.07,1}{CatVar}};
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