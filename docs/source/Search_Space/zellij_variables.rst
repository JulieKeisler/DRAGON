.. _var:

============
Variables
============

**DRAGON** search space is based on :ref:`var`, retrieved from the `zellij <https://zellij.readthedocs.io/en/latest/>`_ package.

******************
Abstract variables
******************

:ref:`var` functionnalities can be extended with :ref:`addons`.

.. automodule:: dragon.search_space.zellij_variables
   :members: Variable
   :undoc-members:
   :show-inheritance:
   :noindex:

**************
Base variables
**************
Basic :ref:`var` are the low level bricks to compose a search space in **DRAGON**.

.. automodule:: dragon.search_space.zellij_variables
   :members: IntVar, FloatVar, CatVar, Constant
   :undoc-members:
   :show-inheritance:
   :noindex:


******************
Composed variables
******************

Composed :ref:`var` are :ref:`var` made of other :ref:`var`.

.. automodule:: dragon.search_space.zellij_variables
   :members: ArrayVar, Block, DynamicBlock
   :undoc-members:
   :show-inheritance:
   :noindex: