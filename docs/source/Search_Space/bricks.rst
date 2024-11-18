.. _candidates:

- Basic Operations

.. automodule:: dragon.search_space.bricks
   :members: Identity, MLP, Dropout
   :undoc-members:
   :show-inheritance:
   :noindex:

- One-dimensional Operations

.. automodule:: dragon.search_space.bricks
   :members: Conv1d, MaxPooling1D, AVGPooling1D, Simple_1DGRU, Simple_1DLSTM, Simple_1DRNN, LayerNorm1d, BatchNorm1d, Attention1D
   :undoc-members:
   :show-inheritance:
   :noindex:

- Two-dimensional Operations

.. automodule:: dragon.search_space.bricks
   :members: Conv2d, MaxPooling2D, AVGPooling2D, Simple_2DGRU, Simple_2DLSTM, LayerNorm2d, BatchNorm2d, SpatialAttention, TemporalAttention
   :undoc-members:
   :show-inheritance:
   :noindex:


- Meta Variables

.. automodule:: dragon.search_space.bricks_variables
   :members: dag_var, node_var, activation_var, operations_var
   :undoc-members:
   :show-inheritance:
   :noindex:

- Basic Variables
.. automodule:: dragon.search_space.bricks_variables
   :members: identity_var, mlp_var, mlp_const_var, dropout
   :undoc-members:
   :show-inheritance:
   :noindex:

- One-dimensional Variables

.. automodule:: dragon.search_space.bricks_variables
   :members: conv_1d, const_conv_1d, pooling_1d, attention_1d, norm_1d, recurrence_1d
   :undoc-members:
   :show-inheritance:
   :noindex:

- Two-dimensional Variables

.. automodule:: dragon.search_space.bricks_variables
   :members: conv_2d, pooling_2d, pooling_2d_const_var, norm_2d, recurrence_2d, attention_2d
   :undoc-members:
   :show-inheritance:
   :noindex: