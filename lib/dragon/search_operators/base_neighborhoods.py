from dragon.search_operators.addons import VarNeighborhood
from dragon.search_space.base_variables import (
    FloatVar,
    IntVar,
    CatVar,
    Constant,
    ArrayVar, 
    DynamicBlock, 
    Block,
    ExclusiveBlock,
    DynamicExclusiveBlock
)
import random
import numpy as np
from dragon.utils.tools import logger
import copy

class IntInterval(VarNeighborhood):
    """IntInterval

    `Addon`, used to determine the neighbor of an IntVar.
    Draw a random point in :math:`x \pm neighborhood`.

    Parameters
    ----------
    variable : IntVar, default=None
        Targeted `Variable`.
    neighborhood : int, default=None
        :math:`x \pm neighborhood`

    Examples
    --------
    >>> from dragon.search_space.base_variables import IntVar
    >>> from dragon.search_operators.base_neighborhoods import IntInterval
    >>> a = IntVar("test", 0, 5, neighbor=IntInterval(neighborhood=1))
    >>> print(a)
    IntVar(test, [0;6])
    >>> a_test = a.random()
    >>> print(a_test)
    4
    >>> a.neighbor(a_test)
    5

    """

    def __call__(self, value, size=1):
        upper = np.min([value + self.neighborhood + 1, self.target.up_bound])
        lower = np.max([value - self.neighborhood, self.target.low_bound])

        if size > 1:
            res = []
            for _ in range(size):
                v = np.random.randint(lower, upper)
                while v == value:
                    v = np.random.randint(lower, upper)
                res.append(int(v))
            return res
        else:
            v = np.random.randint(lower, upper)
            while v == value:
                v = np.random.randint(lower, upper)

            return v

    @VarNeighborhood.neighborhood.setter
    def neighborhood(self, neighborhood):
        assert isinstance(neighborhood, int) or isinstance(
            neighborhood, float
        ), logger.error(
            f"`neighborhood` must be an int, for `IntInterval`,\
            got{neighborhood}"
        )

        self._neighborhood = neighborhood

    @VarNeighborhood.target.setter
    def target(self, variable):

        assert isinstance(variable, IntVar) or variable == None, logger.error(
            f"Target object must be a `IntInterval` for {self.__class__.__name__},\
             got {variable}"
        )
        self._target = variable

class FloatInterval(VarNeighborhood):
    """FloatInterval

    `Addon`, used to determine the neighbor of a FloatVar.
    Draw a random point in :math:`x \pm neighborhood`.

    Parameters
    ----------
    variable : FloatVar, default=None
        Targeted `Variable`.
    neighborhood : float, default=None
        :math:`x \pm neighborhood`

    Examples
    --------
    >>> from dragon.search_space.base_variables import FloatVar
    >>> from dragon.search_operators.base_neighborhoods import FloatInterval
    >>> a = FloatVar("test", 0, 5, neighbor=FloatInterval(neighborhood=1))
    >>> print(a)
    FloatVar(test, [0;5])
    >>> a_test = a.random()
    >>> print(a_test)
    4.0063806879878925
    >>> a.neighbor(a_test)
    4.477278307217116
    """

    def __call__(self, value, size=1):
        upper = np.min([value + self.neighborhood, self.target.up_bound])
        lower = np.max([value - self.neighborhood, self.target.low_bound])

        if size > 1:
            res = []
            for _ in range(size):
                v = np.random.uniform(lower, upper)
                while v == value:
                    v = np.random.uniform(lower, upper)
                res.append(float(v))
            return res
        else:
            v = np.random.uniform(lower, upper)
            while v == value:
                v = np.random.uniform(lower, upper)

            return v

    @VarNeighborhood.neighborhood.setter
    def neighborhood(self, neighborhood):
        assert isinstance(neighborhood, int) or isinstance(
            neighborhood, float
        ), logger.error(
            f"`neighborhood` must be a float or an int, for `FloatInterval`,\
            got{neighborhood}"
        )

        self._neighborhood = neighborhood

    @VarNeighborhood.target.setter
    def target(self, variable):

        assert isinstance(variable, FloatVar) or variable == None, logger.error(
            f"Target object must be a `FloatVar` for {self.__class__.__name__},\
             got {variable}"
        )
        self._target = variable

class CatInterval(VarNeighborhood):
    """CatInterval

    `Addon`, used to determine the neighbor of a CatVar.
    Draw a random feature in CatVar.

    Parameters
    ----------
    variable : CatVar, default=None
        Targeted `Variable`.
    neighborhood : int, default=None
        Undefined, for CatVar it draws a random feature.

    Examples
    --------
    >>> from dragon.search_space.base_variables import CatVar, IntVar
    >>> from dragon.search_operators.base_neighborhoods import CatInterval, IntInterval
    >>> a = CatVar("test", ['a', 1, 2.56, IntVar("int", 100 , 200, neighbor=IntInterval(10))], neighbor=CatInterval())
    >>> print(a)
    CatVar(test, ['a', 1, 2.56, IntVar(int, [100;201])])
    >>> a.neighbor(120, 10) # 10 neighbors for the value '120' within this search space
    [188, 2.56, 'a', 1, 'a', 1, 2.56, 151, 151, 1]
    """

    def __init__(self, variable=None, neighborhood=None):
        super(CatInterval, self).__init__(variable)
        self.neighborhood = neighborhood

    def __call__(self, value, size=1):
        if size > 1:
            res = []
            for _ in range(size):
                v = self.target.random()
                while v == value:
                    v = self.target.random()
                res.append(v)
            return res
        else:
            v = self.target.random()
            while v == value:
                v = self.target.random()
            return v

    @VarNeighborhood.neighborhood.setter
    def neighborhood(self, neighborhood=None):
        if neighborhood != None:
            logger.warning(
                f"`neighborhood`= {neighborhood} is useless for \
            {self.__class__.__name__}, it will be replaced by None"
            )

        self._neighborhood = None

    @VarNeighborhood.target.setter
    def target(self, variable):

        assert isinstance(variable, CatVar) or variable == None, logger.error(
            f"Target object must be a `CatInterval` for {self.__class__.__name__},\
             got {variable}"
        )
        self._target = variable

class ConstantInterval(VarNeighborhood):
    """ConstantInterval

    `Addon`, used to determine the neighbor of a Constant.
    Do nothing. Return the constant.

    Parameters
    ----------
    variable : Constant, default=None
        Targeted `Variable`.

    Examples
    --------
    >>> from dragon.search_space.base_variables import Constant
    >>> from dragon.search_operators.base_neighborhoods import ConstantInterval
    >>> a = Constant("test", 5, neighbor=ConstantInterval())
    >>> print(a)
    Constant(test, 5)
    >>> a_test = a.random()
    >>> print(a_test)
    5
    >>> a.neighbor(a_test)
    5
    """

    def __init__(self, variable=None, neighborhood=None):
        super(ConstantInterval, self).__init__(variable)
        self.neighborhood = neighborhood

    def __call__(self, value, size=1):
        if size > 1:
            return [self.target.value for _ in range(size)]
        else:
            return self.target.value

    @VarNeighborhood.neighborhood.setter
    def neighborhood(self, neighborhood=None):
        if neighborhood != None:
            logger.warning(
                f"`neighborhood`= {neighborhood} is useless for \
            {self.__class__.__name__}, it will be replaced by None"
            )

        self._neighborhood = None

    @VarNeighborhood.target.setter
    def target(self, variable):

        assert isinstance(variable, Constant) or variable == None, logger.error(
            f"Target object must be a `ConstantInterval` for {self.__class__.__name__}\
            , got {variable}"
        )
        self._target = variable

class ArrayInterval(VarNeighborhood):
    """ArrayInterval

    `Addon`, used to determine the neighbor of an ArrayVar.
    neighbor kwarg must be implemented for all `Variable` of the ArrayVar.
    One `Variable` is modified for each neighbor drawn.

    Parameters
    ----------
    variable : ArrayVar, default=None
        Targeted `Variable`.

    Examples
    ----------
    >>> from dragon.search_space.base_variables import ArrayVar, IntVar, FloatVar, CatVar
    >>> from dragon.search_operators.base_neighborhoods import IntInterval, FloatInterval, CatInterval, ArrayInterval
    >>> a = ArrayVar(IntVar("int_1", 0,8, neighbor=IntInterval(2)), IntVar("int_2", 4,45, neighbor=IntInterval(10)), 
    ...              FloatVar("float_1", 2,12, neighbor=FloatInterval(0.5)), CatVar("cat_1", ["Hello", 87, 2.56], neighbor=CatInterval()), neighbor=ArrayInterval())
    >>> print(a)
    ArrayVar(, [IntVar(int_1, [0;9]),IntVar(int_2, [4;46]),FloatVar(float_1, [2;12]),CatVar(cat_1, ['Hello', 87, 2.56])])
    >>> a_test = a.random()
    >>> print(a_test)
    [7, 25, 7.631003022147808, 87]
    >>> a.neighbor(a_test, 10)
    [[7, 25, 8.003980345265523, 87], [8, 25, 7.631003022147808, 87], [7, 25, 7.631003022147808, 2.56], 
    [8, 25, 7.631003022147808, 87], [7, 25, 7.631003022147808, 'Hello'], [7, 17, 7.631003022147808, 87], 
    [7, 25, 7.631003022147808, 2.56], [7, 25, 7.254907155441848, 87], [7, 25, 7.602659938485088, 87], 
    [7, 25, 7.631003022147808, 'Hello']]

    """

    def __init__(self, neighborhood=None, variable=None):
        super(ArrayInterval, self).__init__(variable)
        self._neighborhood = neighborhood

    def __call__(self, value, size=1):
        values = list(self._target.values)
        for v in self._target.values:
            if isinstance(v, Constant):
                values.remove(v)
        variables = np.random.choice(values, size=size)
        if size == 1:
            v = variables[0]
            inter = copy.deepcopy(value)
            inter[v._idx] = v.neighbor(value[v._idx])
            return inter
        else:
            res = []
            for v in variables:
                inter = copy.deepcopy(value)
                inter[v._idx] = v.neighbor(value[v._idx])
                res.append(inter)
            return res

    @VarNeighborhood.neighborhood.setter
    def neighborhood(self, neighborhood=None):
        if neighborhood:
            for var, neig in zip(self._target.values, neighborhood):
                var.neighborhood = neig

        self._neighborhood = None

    @VarNeighborhood.target.setter
    def target(self, variable):

        assert isinstance(variable, ArrayVar) or variable is None, logger.error(
            f"Target object must be an `ArrayVar` for {self.__class__.__name__},\
             got {variable}"
        )

        self._target = variable

        if variable != None:
            assert all(
                hasattr(v, "neighbor") for v in self._target.values
            ), logger.error(
                f"To use `ArrayInterval`, values in `ArrayVar` must have a `neighbor` method. Use `neighbor` kwarg "
                f"when defining a variable "
            )

class BlockInterval(VarNeighborhood):
    """BlockInterval

    `Addon`, used to determine the neighbor of an BlockInterval.
    neighbor kwarg must be implemented for all `Variable` of the BlockInterval.
    
    Parameters
    ----------
    variable : Block, default=None
        Targeted `Variable`.

    Examples
    ----------
    >>> from dragon.search_space.base_variables import Block, ArrayVar, FloatVar, IntVar
    >>> from dragon.search_operators.base_neighborhoods import BlockInterval, ArrayInterval, FloatInterval, IntInterval
    >>> content = ArrayVar(IntVar("int_1", 0,8, neighbor=IntInterval(2)), IntVar("int_2", 4,45, neighbor=IntInterval(10)),  FloatVar("float_1", 2,12, neighbor=FloatInterval(10)), neighbor=ArrayInterval())
    >>> a = Block("max size 10 Block", content, 3, neighbor=BlockInterval())
    >>> print(a)
    Block(max size 10 Block, [IntVar(int_1, [0;9]),IntVar(int_2, [4;46]),FloatVar(float_1, [2;12]),])
    >>> test_a = a.random()
    >>> print(test_a)
    [[5, 4, 10.780991223247005], [1, 11, 11.446866387945619], [8, 44, 2.9377647083768217]]
    >>> a.neighbor(test_a)
    [[5, 7, 10.780991223247005], [0, 11, 11.446866387945619], [8, 44, 2.9377647083768217]]
    """

    def __init__(self, neighborhood=None, variable=None):
        self._neighborhood = neighborhood
        super(BlockInterval, self).__init__(variable)

    def __call__(self, value, size=1):
        if size == 1:
            res = copy.deepcopy(value)
            variables_idx = list(set(np.random.choice(range(self.target.repeat), size=self.target.repeat)))
            for i in variables_idx:
                res[i] = self.target.value.neighbor(value[i])
        else:
            res = []
            for _ in range(size):
                inter = copy.deepcopy(value)
                variables_idx = list(set(np.random.choice(range(self.target.repeat), size=self.target.repeat)))
                for i in variables_idx:
                    inter[i] = self.target.value.neighbor(value[i])
                res.append(inter)
        return res

    @VarNeighborhood.neighborhood.setter
    def neighborhood(self, neighborhood=None):
            self._neighborhood = neighborhood

    @VarNeighborhood.target.setter
    def target(self, variable):
        assert isinstance(variable, Block) or variable is None, logger.error(
            f"Target object must be a `Block` for {self.__class__.__name__},\
             got {variable}"
        )
        self._target = variable

        if variable is not None:
            assert hasattr(self._target.value, "neighbor"), logger.error(
                f"To use `Block`, value for `Block` must have a `neighbor` method. Use `neighbor` kwarg "
                f"when defining a variable "
            )

class DynamicBlockInterval(VarNeighborhood):
    """BlockInterval

    `Addon`, used to determine the neighbor of a DynamicBlock.
    neighbor kwarg must be implemented for all `Variable` of the BlockInterval.

    Parameters
    ----------
    variable : IntVar, default=None
        Targeted `Variable`.
    neighborhood : int
        Neighborhood of the DynamicBlock size

    Example
    ----------
    
    >>> from dragon.search_space.base_variables import DynamicBlock, ArrayVar, FloatVar, IntVar
    >>> from dragon.search_operators.base_neighborhoods import DynamicBlockInterval, ArrayInterval, FloatInterval, IntInterval
    >>> content = ArrayVar(IntVar("int_1", 0,8, neighbor=IntInterval(2)), IntVar("int_2", 4,45, neighbor=IntInterval(10)),  FloatVar("float_1", 2,12, neighbor=FloatInterval(10)), neighbor=ArrayInterval())
    >>> a = DynamicBlock("max size 10 Block", content, 5, neighbor=DynamicBlockInterval(1))
    >>> print(a)
    DynamicBlock(max size 10 Block, [IntVar(int_1, [0;9]),IntVar(int_2, [4;46]),FloatVar(float_1, [2;12]),])
    >>> test_a = a.random()
    >>> print(test_a)
    [[4, 10, 7.476654992446498]]
    >>> a.neighbor(test_a)
    [[4, 17, 7.476654992446498], [2, 5, 8.057170687346623], [2, 19, 7.316509989314727], [8, 9, 8.294482483654278], [2, 31, 5.36321423474537]]

    """

    def __call__(self, value, size=1, new_repeat=None):
        res = []
        for _ in range(size):
            if new_repeat is None:
                new_repeat = np.random.randint(self.target.repeat - self._neighborhood,
                                           self.target.repeat + self._neighborhood+1)
            inter = copy.deepcopy(value)
            if new_repeat > len(inter):
                inter+=[l if (new_repeat - len(inter))==1 else l[0] for l in self.target.random(new_repeat - len(inter))]
            if new_repeat < len(inter):
                deleted_idx = list(set(random.sample(range(len(inter)), len(inter) - new_repeat)))
                for index in sorted(deleted_idx, reverse=True):
                    del inter[index]
            variables_idx = list(set(np.random.choice(range(new_repeat), size=new_repeat)))
            for i in variables_idx:
                inter[i] = self.target.value.neighbor(inter[i])
            res.append(inter)
        if size == 1:
            return res[0]
        else:
            return res

    @VarNeighborhood.neighborhood.setter
    def neighborhood(self, neighborhood=None):
        if isinstance(neighborhood, list):
            self._neighborhood = neighborhood[0]
            self.target.value.neighborhood = neighborhood[1]
        else:
            self._neighborhood = neighborhood

    @VarNeighborhood.target.setter
    def target(self, variable):
        assert isinstance(variable, DynamicBlock) or variable is None, logger.error(
            f"Target object must be a `DynamicBlock` for {self.__class__.__name__},\
             got {variable}"
        )
        self._target = variable

        if variable is not None:
            assert hasattr(self._target.value, "neighbor"), logger.error(
                f"To use `DynamicBlock`, value for `DynamicBlock` must have a `neighbor` method. Use `neighbor` kwarg "
                f"when defining a variable "
            )

class ExclusiveBlockInterval(VarNeighborhood):
    """ExclusiveBlockInterval

    `Addon`, used to determine the neighbor of an ExclusiveBlock.
    neighbor kwarg must be implemented for all `Variable` of the ExclusiveBlockInterval.

    Parameters
    ----------
    variable : IntVar, default=None
        Targeted `Variable`.

    Example
    ----------
    >>> from dragon.search_space.base_variables import ExclusiveBlock, ArrayVar, FloatVar, IntVar
    >>> from dragon.search_operators.base_neighborhoods import ExclusiveBlockInterval, ArrayInterval, FloatInterval, IntInterval
    >>> content = ArrayVar(IntVar("int_1", 0,8, neighbor=IntInterval(2)), IntVar("int_2", 4,45, neighbor=IntInterval(10)),  FloatVar("float_1", 2,12, neighbor=FloatInterval(10)), neighbor=ArrayInterval())
    >>> a = ExclusiveBlock("max size 10 Block", content, 5, neighbor=ExclusiveBlockInterval())
    >>> print(a)
    ExclusiveBlock(max size 10 Block, [IntVar(int_1, [0;9]),IntVar(int_2, [4;46]),FloatVar(float_1, [2;12]),])
    >>> test_a = a.random()
    >>> print(test_a)
    [[2, 25, 3.407434018008985], [8, 34, 11.720825933953947], [4, 7, 5.294945631972848], [3, 17, 11.621101715902546], [5, 24, 3.194865279405992]]
    >>> a.neighbor(test_a)
    [[2, 25, 3.407434018008985], [8, 34, 11.720825933953947], [5, 7, 5.294945631972848], [5, 17, 11.621101715902546], [5, 22, 3.194865279405992]]
    """
    def __init__(self, neighborhood=None, variable=None):
        self.neighborhood = neighborhood
        super(ExclusiveBlockInterval, self).__init__(variable)

    def __call__(self, value, size=1):
        res = []
        for _ in range(size):
            inter = copy.deepcopy(value)
            variables_idx = sorted(list(set(np.random.choice(range(self.target.repeat), size=self.target.repeat))))
            for i in variables_idx:
                inter[i] = self.target.value.neighbor(value[i])
                
                #Verification of the unicity of each value
                while (inter[i] in inter[:i]):
                    inter[i] = self.target.value.neighbor(inter[i])

                res.append(inter)

        if (size == 1):
            return res[0]
        else:
            return res

    @VarNeighborhood.neighborhood.setter
    def neighborhood(self, neighborhood=None):
            self._neighborhood = neighborhood

    @VarNeighborhood.target.setter
    def target(self, variable):
        assert(
            isinstance(variable, ExclusiveBlock) or variable is None
        ), f"""Target object must be a `Block` for {self.__class__.__name__},\
        got {variable}."""
        
        self._target = variable

        if variable is not None:
            assert(
                hasattr(self._target.value, "neighbor")
            ),  f"""To use `ExclusiveBlock`, value for `ExclusiveBlock` must have a `neighbor` method. Use `neighbor` kwarg when defining a variable."""


class DynamicExclusiveBlockInterval(VarNeighborhood):
    """DynamicExclusiveBlockInterval

    `Addon`, used to determine the neighbor of an ExclusiveDynamicBlock.
    neighbor kwarg must be implemented for all `Variable` of the ExclusiveDynamicBlockInterval.

    Parameters
    ----------
    variable : IntVar, default=None
        Targeted `Variable`.
    neighborhood : int
        Neighborhood of the ExclusiveDynamicBlock size

    Example
    ----------
    >>> from dragon.search_space.base_variables import DynamicExclusiveBlock, ArrayVar, FloatVar, IntVar
    >>> from dragon.search_operators.base_neighborhoods import DynamicExclusiveBlockInterval, ArrayInterval, FloatInterval, IntInterval
    >>> content = ArrayVar(IntVar("int_1", 0,8, neighbor=IntInterval(2)), IntVar("int_2", 4,45, neighbor=IntInterval(10)),  FloatVar("float_1", 2,12, neighbor=FloatInterval(10)), neighbor=ArrayInterval())
    >>> a = DynamicExclusiveBlock("max size 10 Block", content, 5, neighbor=DynamicExclusiveBlockInterval(1))
    >>> print(a)
    DynamicExclusiveBlock(max size 10 Block, [IntVar(int_1, [0;9]),IntVar(int_2, [4;46]),FloatVar(float_1, [2;12]),])
    >>> test_a = a.random()
    >>> print(test_a)
    [[4, 42, 10.450686412997023], [1, 37, 7.027430133368101]]
    >>> a.neighbor(test_a)
    [[4, 42, 10.450686412997023], [1, 29, 7.027430133368101], [5, 5, 8.421363070710674]]
    """
    def __call__(self, value, size=1, new_repeat=None):
        res = []
        for _ in range(size):
            inter = copy.deepcopy(value)

            if new_repeat is None:
                #If a new length for value has not explicitly been given,
                #we choose a new one as a function of the given neighborhood
                #while remaining in the given length limits
                new_repeat = np.random.randint(max(len(inter) - self._neighborhood, self.target.min_repeat),
                                                min(len(inter) + self._neighborhood, self.target.repeat) + 1)

            #Adding new unique coefficients if the new length is greater
            if new_repeat > len(inter):
                diff = new_repeat - len(inter)
                for _ in range(diff):
                    add_coef = self.target.value.random()
                    while add_coef in inter:
                        add_coef = self.target.value.random()
                    inter.append(add_coef)

            #Removing values at random if the new length is smaller
            if new_repeat < len(inter):
                deleted_idx = np.random.choice(range(len(inter)), size = len(inter)-new_repeat, replace=False)
                for index in sorted(deleted_idx, reverse=True):
                    del inter[index]

            #The following line chooses value indices to be mutated
            variables_idx = sorted(list(set(np.random.choice(range(new_repeat), size=new_repeat))))

            for i in variables_idx:
                inter[i] = self.target.value.neighbor(inter[i])

                #Verification of the unicity of each value
                while (inter[i] in inter[:i]):
                    inter[i] = self.target.value.neighbor(inter[i])

            res.append(inter)

        if size == 1:
            return res[0]
        else:
            return res

    @VarNeighborhood.neighborhood.setter
    def neighborhood(self, neighborhood=None):
        if isinstance(neighborhood, list):
            self._neighborhood = neighborhood[0]
            self.target.value.neighborhood = neighborhood[1]
        else:
            self._neighborhood = neighborhood

    @VarNeighborhood.target.setter
    def target(self, variable):
        assert(
            isinstance(variable, DynamicExclusiveBlock) or variable is None
        ), f"""Target object must be a `DynamicExclusiveBlock` for {self.__class__.__name__}, got {variable}."""

        self._target = variable

        if variable is not None:
            assert(
                hasattr(self._target.value, "neighbor")
            ), f"""To use `DynamicExclusiveBlock`, value for `DynamicExclusiveBlock` must have a `neighbor` method. Use `neighbor` kwarg when defining a variable."""

