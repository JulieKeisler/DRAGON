from abc import ABC, abstractmethod
from dragon.utils.tools import logger

class Addon(ABC):
    """Addon

    Abstract class describing what an addon is.
    An :code:`Addon` is an additionnal feature that can be added to a
    :code:`target` object. See :ref:`varadd` for addon targeting :ref:`var` or
    :ref:`spadd` targeting :ref:`sp`.

    Parameters
    ----------
    target : Object, default=None
        Object targeted by the addons

    Attributes
    ----------
    target : Object, default=None
        Object targeted by the addons

    """

    def __init__(self, object=None):
        self.target = object

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, object):
        self._target = object


class VarAddon(Addon):
    """VarAddon

    :ref:`addons` where the target must be of type :ref:`var`.

    Parameters
    ----------
    target : :ref:`var`, default=None
        Object targeted by the addons

    Attributes
    ----------
    target : :ref:`var`, default=None
        Object targeted by the addons

    """

    def __init__(self, variable=None):
        super(VarAddon, self).__init__(variable)

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, variable):
        from dragon.search_space.zellij_variables import Variable

        if variable:
            assert isinstance(variable, Variable), logger.error(
                f"Object must be a `Variable` for {self.__class__.__name__}, got {variable}"
            )

        self._target = variable

class SearchspaceAddon(Addon):
    """SearchspaceAddon

    :ref:`addons` where the target must be of type :ref:`sp`.

    Parameters
    ----------
    target : :ref:`sp`, default=None
        Object targeted by the addons

    Attributes
    ----------
    target : :ref:`sp`, default=None
        Object targeted by the addons

    """

    def __init__(self, search_space=None):
        super(SearchspaceAddon, self).__init__(search_space)

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, search_space):
        self._target = search_space

class Neighborhood(SearchspaceAddon):
    """Neighborhood

    :ref:`addons` where the target must be of type :ref:`sp`.
    Describes what a neighborhood is for a :ref:`sp`.

    Parameters
    ----------
    target : :ref:`sp`, default=None
        Object targeted by the addons

    Attributes
    ----------
    target : :ref:`sp`, default=None
        Object targeted by the addons

    """

    def __init__(self, neighborhood, search_space=None):
        super(Neighborhood, self).__init__(search_space)
        self.neighborhood = neighborhood

    @property
    def neighborhood(self):
        return self._neighborhood

    @abstractmethod
    def __call__(self, point, size=1):
        pass


class VarNeighborhood(VarAddon):
    """VarNeighborhood

    :ref:`addons` where the target must be of type :ref:`var`.
    Describes what a neighborhood is for a :ref:`var`.

    Parameters
    ----------
    target : :ref:`var`, default=None
        Object targeted by the addons

    Attributes
    ----------
    target : :ref:`var`, default=None
        Object targeted by the addons

    """

    def __init__(self, neighborhood, variable=None):
        super(VarAddon, self).__init__(variable)
        self.neighborhood = neighborhood

    @property
    def neighborhood(self):
        return self._neighborhood

    @abstractmethod
    def __call__(self, point, size=1):
        pass

class Operator(SearchspaceAddon):
    """Operator

    Abstract class describing what an operator is for a :ref:`sp`.
    :ref:`addons` where the target must be of type :ref:`sp`.

    Parameters
    ----------
    target : :ref:`sp`, default=None
        Object targeted by the addons

    Attributes
    ----------
    target : :ref:`sp`, default=None
        Object targeted by the addons

    """

    def __init__(self, search_space=None):
        super(Operator, self).__init__(search_space)

    @abstractmethod
    def __call__(self):
        pass


class Mutator(SearchspaceAddon):
    """Mutator

    Abstract class describing what an Mutator is for a :ref:`sp`.
    :ref:`addons` where the target must be of type :ref:`sp`.

    Parameters
    ----------
    target : :ref:`sp`, default=None
        Object targeted by the addons

    Attributes
    ----------
    target : :ref:`sp`, default=None
        Object targeted by the addons

    """

    def __init__(self, search_space=None):
        super(Mutator, self).__init__(search_space)


class Crossover(SearchspaceAddon):
    """Crossover

    Abstract class describing what an MCrossover is for a :ref:`sp`.
    :ref:`addons` where the target must be of type :ref:`sp`.

    Parameters
    ----------
    target : :ref:`sp`, default=None
        Object targeted by the addons

    Attributes
    ----------
    target : :ref:`sp`, default=None
        Object targeted by the addons

    """

    def __init__(self, search_space=None):
        super(Crossover, self).__init__(search_space)