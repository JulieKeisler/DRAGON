.. _search_space:

=============================
Presentation
=============================

The search space design is based on an abstract class called *Variable*, originally proposed within an hyper-parameters optimization package called `zellij <https://zellij.readthedocs.io/en/latest/>`_.
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
   print(a)
   ArrayVar(, [IntVar(int_1, [0;8]),
                IntVar(int_2, [4;45]),
                FloatVar(float_1, [2;12]),
                CatVar(cat_1, ['Hello', 87, 2.56])])

   a.random()
   [5, 15, 8.483221226216427, 'Hello']

Here we created an array made of four distinct elements: two integers respectively between 0 and 8 and 4 and 45, a float between 2 and 12 and a categorical variable which can take values within ["Hello", 87, 2.56].
In opposition to the `CatVar` features attribute which might mix variables and non-variables elements, the attribute from the composed variables have to be variables. It means we cannot create an array with a simple 5. To inclue a constant integer, we have to pass through the `Constant` variable.

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

DAGs Encoding
------------

Both the base and composed variables have been used to encode Deep Neural Networks architecture and hyper-parameters.

Operations and hyperparameters encoding
~~~~~~~~~~~~~~~~~~~~~~

The Deep Neural Networks are made of layers. In **DRAGON**'s case, those layers are *nn.Module* from *PyTorch*.
The user can use any base or custom *nn.Module*, but as to wrap it into a *Brick* object. 
A brick takes as input an input shape and some hyper-parameters and initialize a given *nn.Module* with these hyperparameters so it can pocess a tensor of the given input shape.
The forward pass of a *Brick* can just apply the layer to an input tensor, or be more complex to transform the input data before the operation.
Finally, the abstract class *Brick* also implements a *modify_operation* method. 
It takes as input an `input_shape` tuple and modifies the operation weights shape, so that the operation may take as input a vector of shape `input_shape`.
This method is applied when the Deep Neural Network is created or modified.
The applications case will be detailled below.

+---------------------------------------------------+-----------------------------------------------------------------------------------------+
|                                                   |                                                                                         |
|.. code-block::                                    |.. code-block::                                                                          |
|                                                   |                                                                                         |
|  import torch.nn as nn                            | from dragon.search_space.cells import Brick                                             |
|  from dragon.search_space.cells import Brick      | import torch.nn as nn                                                                   |
|                                                   |                                                                                         |
|  class Dropout(Brick):                            | class MLP(Brick):                                                                       |
|     def __init__(self, input_shape, rate):        |    def __init__(self, input_shape, out_channels):                                       |
|        super(Dropout, self).__init__(input_shape) |       super(MLP, self).__init__(input_shape)                                            |
|        self.dropout = nn.Dropout(p=rate)          |       self.in_channels = input_shape[-1]                                                |
|     def forward(self, X):                         |       self.linear = nn.Linear(self.in_channels, out_channels)                           |
|        X = self.dropout(X)                        |    def forward(self, X):                                                                |
|        return X                                   |       X = self.linear(X)                                                                |
|     def modify_operation(self, input_shape):      |       return X                                                                          |
|        pass                                       |    def modify_operation(self, input_shape):                                             |
|                                                   |       d_in = input_shape[-1]                                                            |
|                                                   |       diff = d_in - self.in_channels                                                    |
|                                                   |       sign = diff / np.abs(diff) if diff !=0 else 1                                     |
|                                                   |       pad = (int(sign * np.ceil(np.abs(diff)/2)), int(sign * np.floor(np.abs(diff))/2)) |
|                                                   |       self.in_channels = d_in                                                           |
|                                                   |       self.linear.weight.data = nn.functional.pad(self.linear.weight, pad)              |
+---------------------------------------------------+-----------------------------------------------------------------------------------------+

The codes just above show respectively the implementation of a `Dropout` and an `MLP` layer. 
While the wrapping of the `Dropout` layer into a `Brick` object requires minimal modifications, the `MLP` wrapping necessitates some effort to implement the `modify_operation` layer.
Indeed, the weights of an `nn.Linear` shape layer depends on the input tensor dimension.

The variable encoding a `Brick` is called `HpVar`.

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
   Operation and hp\\=\\\textcolor[rgb]{0.56,0.07,1}{HpVar}
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