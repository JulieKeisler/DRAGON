=============================
Performance evaluation
=============================

**DRAGON** can be used for a wide range of applications. 
To apply the package to a particular task, one needs to:
* Create a suitable search space with the variables introduced in `Search Space <../Search_Space/index.rst>`_.
* Select (or implement) a search algorithm such as those presented in `Search Algorithm <../Search_Algorithm/index.rst>`_. Depending on the type of search algorithm, it will be necessary to associate `neighbor` with the variables in the search space, as described in section `Search Operators <../Search_Operators/index.rst>`_.
* Implement a performance evaluation function.

This function should take as input a configuration (and the index of that configuration in the case of algorithms implemented in the package), and return a `loss`` corresponding to the evaluation of the configuration. 
The `loss` and the way it is calculated depends on the task at hand. 
In general, it is necessary to build a neural network from the configuration, train it and validate it on data sets.
Let's take the example of a problem classifying a vector :math:`X \in \mathbb{R}^n`, with :math:`n \in \mathbb{N}^\star` into 10 different classes.
We want to find the best performing model among any type of architecture that takes an input of size `n` to an output of size 10. 
This type of architecture can be represented by a DAG.
However, in order to constrain the output to be of dimension 10, it is necessary to add a final layer that allows the output of any size from the DAG to be converted to a vector of size 10.
The DAG and the output layer are associated together within a `nn.Module` called `MetaArchi`.

.. code-block:: python

   class MetaArchi(nn.Module):
    def __init__(self, args, input_shape):
        super().__init__()
        # Number of features, here equals to n
        self.input_shape = input_shape

        # The Neural Network will be optimised through a DAG
        self.dag = args['Dag']
        self.dag.set(input_shape)

        # We set the final layer
        self.output = args["Out"]
        self.output.set(self.dag.output_shape)

    def forward(self, X):
        out = self.dag(X)
        return self.output(out)

The class `MetaArchi` takes as arguments the variable `args` which contain the configuration indicating how to build the model and `input\_shape` indicating the shape of the tensors that will be processed.
The argument `self.dag` is an `AdjMatrix` and `self.output` is a `Node`.
Their methods `set` were explained in the `Search Space section <../Search_Space/index.rst>`_.
They adjust the `nn.Module` weights to ensure they can handle tensors with the right input shape.
The search space used to create the variable `args` is specified by the user.
In the following example, we create a search space made of a DAG only having `MLP` and `Identity` layers as candidate operations.

.. code-block:: python

   from dragon.search_space.bricks_variables import mlp_var, identity_var, operations_var, mlp_const_var, dag_var, node_var
   from dragon.search_space.base_variables import ArrayVar
   from dragon.search_operators.base_neighborhoods import ArrayInterval

   # Candidate operations for the DAG: MLP and Identity layers.
   candidate_operations = operations_var("Candidate operations", size=10, candidates=[mlp_var("MLP"), identity_var("Identity")])
   dag = dag_var("Dag", candidate_operations)
   # Output layer with a Softmax activation function for the classification
   out = node_var("Out", operation=mlp_const_var('Operation', 10), activation_function=nn.Softmax())
   # Global search space
   search_space = ArrayVar(dag, out, label="Search Space", neighbor=ArrayInterval())

Finally, the last step is to implement a training and a validation strategy to assess the performance of a configuration from the search space.
This can be done by splitting the available dataset into a train and validation set.

.. code-block:: python

   def train_model(model, data_loader):
      # Model training through gradient descent
      loss_fn = nn.CrossEntropyLoss()
      optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
      model.train()
      for _ in range(2):
         for X,y in data_loader:
               optimizer.zero_grad()
               y = y.squeeze()
               pred = model(X)
               loss = loss_fn(pred,y)
               loss.backward()
               optimizer.step()
      return model

   def validate_model(model, data_loader):
      # Compute the prediction of the trained model on a validation set
      loss_fn = nn.CrossEntropyLoss()
      model.eval()
      test_loss, correct = 0, 0
      with torch.no_grad():
         for X, y in data_loader:
            y = y.squeeze(1)
            pred = model(X)
            loss = loss_fn(pred, y).item()
            test_loss += loss
            prediction = pred.argmax(axis=1)
            correct += (prediction == y).sum().item()
      accuracy = correct/ len(data_loader.dataset)
      return accuracy

   def loss_function(args, idx, *kwargs):
      labels = [e.label for e in search_space]
      args = dict(zip(labels, args))
      model = MetaArchi(args, input_shape=(16,))
      model = train_model(model, train_loader)
      accuracy = test_model(model, val_loader)
      print(f"Idx = {idx}, loss = {1-accuracy}")
      # Return the loss
      return 1 - accuracy, model

   loss, model = loss_function(search_space.random())

The function `loss_function` can be passed to any **DRAGON** search algorithm as the `evaluation` function.
The `search_space` object can be passed as the `search_space` argument.

.. code-block:: python

   from dragon.search_algorithm.ssea import SteadyStateEA

   search_algorithm = SteadyStateEA(search_space, n_iterations=20, population_size=5, selection_size=3, evaluation=loss_function, save_dir="save/test_image/")
   search_algorithm.run()

For more detailed examples of various applications, see:

.. toctree::
   :maxdepth: 1

   image
   load_forecasting
