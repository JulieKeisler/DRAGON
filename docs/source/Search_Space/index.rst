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
::
   3

In this example, the variable `v` defines an integer which can take values from 0 to 5. When calling `v.random()`, the script returns an integer from this interval, here `3`.
All the base variables available within **DRAGON** are detailed in the `Base Variables <_base_variables>`_ section.


Composed variables
------------

The base variables can be composed to create more complex objects such as arrays of variables.

These fundamental elements have been leveraged within the **DRAGON** package to generate new tools for optimizing both the architecture and the hyperparameters of deep neural networks. These tools are very generic and allow the user to use any `nn.Module` object within the optimized architectures. Some basic operations are already implemented and ready to use to facilitate the use of the package.

DAG Encoding
------------

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
   \draw (229,282) node [anchor=north west][inner sep=0.75pt]   [align=left] {\textcolor[rgb]{0.29,0.56,0.89}{Brick} or list of \textcolor[rgb]{0.29,0.56,0.89}{Bricks }(\textit{PyTorch} operation)\textcolor[rgb]{0.29,0.56,0.89}{ }= \textcolor[rgb]{0.56,0.07,1}{Constant}\textcolor[rgb]{0.29,0.56,0.89}{ }or\textcolor[rgb]{0.29,0.56,0.89}{ }\textcolor[rgb]{0.56,0.07,1}{CatVar}};
   % Text Node
   \draw (229,324) node [anchor=north west][inner sep=0.75pt]   [align=left] {Hyperparameters = dictionnary of base variables \\(e.g \textcolor[rgb]{0.56,0.07,1}{FloatVar}, \textcolor[rgb]{0.56,0.07,1}{CatVar})};
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