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
        from dragon.search_space.base_variables import Variable

        if variable:
            assert isinstance(variable, Variable), logger.error(
                f"Object must be a `Variable` for {self.__class__.__name__}, got {variable}"
            )

        self._target = variable

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