import numpy as np
from zellij.core.addons import Mutator
from zellij.core.search_space import Searchspace
from zellij.core.variables import Constant

from dragon.search_space.dags import AdjMatrixVariable


class HierarchicalNNMutation(Mutator):
    def __init__(self, probability, search_space=None, change=30, idx_min=0, idx_max=4):
        assert (
                0 < probability <= 1
        ), f'Probability must be comprised in ]0,1], got {probability}'
        self.probability = probability
        if isinstance(change, int):
            change = ['large' for _ in range(change * 10)] + ['local' for _ in range(change)]
        self.change = change
        self.idx_min = idx_min
        self.idx_max = idx_max

        super(Mutator, self).__init__(search_space)

    @Mutator.target.setter
    def target(self, search_space):
        if search_space:
            assert isinstance(
                search_space, Searchspace
            ), f""" Target object must be a :ref:`sp`
                for {self.__class__.__name__}, got {search_space}"""

            assert all(
                hasattr(val, "neighbor") for val in search_space.values
            ), f"""For {self.__class__.__name__} values of target object must
                have a `neighbor` method.
                """
        self._target = search_space

    def _build(self, toolbox):
        toolbox.register("mutate", self, toolbox.generation)

    def __call__(self, generation, individual):
        # For each dimension of a solution draw a probability to be muted
        g = generation()
        try:
            neigh = self.change[g]
        except IndexError:
            neigh = self.change[-1]
        for val in self.target.values:
            if np.random.random() < self.probability and not isinstance(
                    val, Constant
            ):
                # Get a neighbor of the selected attribute
                if isinstance(val, AdjMatrixVariable):
                    individual[val._idx] = val.neighbor(individual[val._idx], neigh=neigh)
                else:
                    individual[val._idx] = val.neighbor(individual[val._idx])
        return individual,


class NNMutation(Mutator):
    def __init__(self, probability, search_space=None, idx_min=0, idx_max=4):
        assert (
                0 < probability <= 1
        ), f'Probability must be comprised in ]0,1], got {probability}'
        self.probability = probability
        self.idx_min = idx_min
        self.idx_max = idx_max

        super(Mutator, self).__init__(search_space)

    @Mutator.target.setter
    def target(self, search_space):
        if search_space:
            assert isinstance(
                search_space, Searchspace
            ), f""" Target object must be a :ref:`sp`
                for {self.__class__.__name__}, got {search_space}"""

            assert all(
                hasattr(val, "neighbor") for val in search_space.values
            ), f"""For {self.__class__.__name__} values of target object must
                have a `neighbor` method.
                """
        self._target = search_space

    def _build(self, toolbox):
        toolbox.register("mutate", self, toolbox.generation)

    def __call__(self, generation, individual):
        # For each dimension of a solution draw a probability to be muted
        for val in self.target.values:
            if np.random.random() < self.probability and not isinstance(
                    val, Constant
            ):
                # Get a neighbor of the selected attribute
                individual[val._idx] = val.neighbor(individual[val._idx])
        return individual,


