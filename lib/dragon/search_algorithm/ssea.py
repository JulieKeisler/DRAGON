from copy import deepcopy
import os
import pickle
import random
from dragon.utils.tools import logger
from dragon.search_operators.crossover import DAGTwoPoint
from dragon.search_algorithm.search_algorithm import SearchAlgorithm
import random
import pandas as pd
import numpy as np

class SteadyStateEA(SearchAlgorithm):
    """SteadyStateEA

    Search algorithm implementing the steady-state (asynchronous) search algorithm inheriting from the `SearchAlgorithm` class.
    It implements a `select_next_configuration` and a `process_evaluated_configuration` methods, specific to the evolutionary algorithm.

    Parameters
    ----------
    search_space: `Variable`
        `Variable` containing all the design choices from the search space. It should implement a `random` method and a `neighbor` one.
    n_iterations: int
        Number of iterations.
    population_size: int
        Size of the population.
    selection_size: int:
        Size of the tournament selection.
    crossover: function, default=DAGTwoPoint()
        Crossover function to use.
    evaluation: function
        Performance evaluation function. Takes as argument a set of configuration and the unique index of this configuration. Returns the performance and the model built.
    save_dir: str
        Path towards saving directory. If not empty, the content will be replaced.
    models: list, default=None
        List of configurations that should be included into the initial population.
    pop_path: str, default=None
        Path towards a directory containing an former evaluation that we aim to continue.
    verbose: bool, default=False
        Verbose boolean.

    Attributes
    ----------
    selection_size: int:
        Size of the tournament selection.
    crossover: function, default=DAGTwoPoint()
        Crossover function to use.
    sent: dict, default={}
        Dictionary containing the configurations that are currently evaluated and thus are temporarly removed from the population.
    search_space: `Variable`
        `Variable` containing all the design choices from the search space. It should implement a `random` method and a `neighbor` one if necessary.
    n_iterations: int
        Number of iterations.
    population_size: int
        Size of the randomly initialized population.
    evaluation: function
        Performance evaluation function. Takes as argument a set of configuration and the unique index of this configuration. Returns the performance and the model built.
    save_dir: str
        Path towards saving directory. If not empty, the content will be replaced.
    models: list, default=None
        List of configurations that should be included into the initial population.
    pop_path: str, default=None
        Path towards a directory containing an former evaluation that we aim to continue.
    verbose: bool, default=False
        Verbose boolean.
    run: function
        Run function to use: MPI (run_mpi) or not (run_no_mpi).
    set_mpi: dict
        Dictionary containing the MPI parameters.
    storage: dict, default={}
        Dictionary storing the configurations from the population.
    min_loss: float, default=np.min
        Current minimum loss found.

    Example
    --------
    >>> from dragon.search_space.base_variables import ArrayVar
    >>> from dragon.search_operators.base_neighborhoods import ArrayInterval
    >>> from dragon.search_algorithm.ssea import SteadyStateEA
    >>> search_space = ArrayVar(dag, label="Search Space", neighbor=ArrayInterval())
    >>> search_algorithm = SteadyStateEA(search_space, save_dir="save/test_ssea", n_iterations=20, population_size=5, selection_size=3, evaluation=loss_function)
    >>> search_algorithm.run()
    """
    def __init__(self, search_space, n_iterations: int, population_size: int, selection_size: int, evaluation, save_dir, models = None, pop_path=None, crossover=DAGTwoPoint(), verbose=False, **args):
        super(SteadyStateEA, self).__init__(search_space=search_space, 
                                            n_iterations=n_iterations, 
                                            init_population_size=population_size, evaluation=evaluation, save_dir=save_dir, models=models, pop_path=pop_path, verbose=verbose)
        self.selection_size = selection_size
        self.crossover = crossover
        

    def select_next_configurations(self):
        """select_next_configurations()
        Defines a selection strategy for SSEA.
        Uses tournament selection of size `self.selection_size` to select two parent configurations.
        Creates two offsprings by mutating (through the `neighbor` attribute of the configuration) and with the crossover function.
        
        Returns
        ----------
        [self.K-1, self.K]: list
            List containing the indexes of the two offsprings.
        """
        not_selected = True
        while not_selected:
            selection = [random.choice(list(self.storage.keys())) for i in range(min(self.selection_size, len(self.storage)))]
            best_1 = selection[np.argmin([self.storage[i]['Loss'] for i in selection])]
            try:
                parent1 = self.storage.pop(best_1)
                with open(f"{self.save_dir}/x_{best_1}.pkl", 'rb') as f:
                    x1 = pickle.load(f)
                selection = [random.choice(list(self.storage.keys())) for i in range(min(self.selection_size, len(self.storage)))]
                not_selected=False
            except Exception as e:
                logger.error(f'Could not load individual {best_1}/{len(list(self.storage.keys()))}, {e}')
        not_selected = True
        while not_selected:
            selection = [random.choice(list(self.storage.keys())) for i in range(min(self.selection_size, len(self.storage)))]
            best_2 = selection[np.argmin([self.storage[i]['Loss'] for i in selection])]
            try:
                parent2 = self.storage.pop(best_2)
                with open(f"{self.save_dir}/x_{best_2}.pkl", 'rb') as f:
                    x2 = pickle.load(f)
                not_selected = False
            except Exception as e:
                logger.error(f'Could not load individual {best_2}/{len(list(self.storage.keys()))}, {e}')
        self.storage[best_1] = parent1
        self.storage[best_2] = parent2
        offspring_1, offspring_2 = deepcopy(x1), deepcopy(x2)
        self.crossover(offspring_1, offspring_2)
        not_muted = True
        while not_muted:
            try:
                offspring_1 = self.search_space.neighbor(deepcopy(offspring_1))
                not_muted = False
            except Exception as e:
                logger.error(f"While mutating, an exception was raised: {e}")
        not_muted = True
        while not_muted:
            try:
                offspring_2 = self.search_space.neighbor(deepcopy(offspring_2))
                not_muted = False
            except Exception as e:
                logger.error(f"While mutating, an exception was raised: {e}")
        with open(f"{self.save_dir}/x_{self.K+1}.pkl", 'wb') as f:
            pickle.dump(offspring_1, f)
        with open(f"{self.save_dir}/x_{self.K+2}.pkl", 'wb') as f:
            pickle.dump(offspring_2, f)
        del offspring_1
        del offspring_2
        logger.info(f"Evolving {best_1} and {best_2} to {self.K+1} and {self.K+2}")
        self.K+=2
        return [self.K-1, self.K]

    def process_evaluated_configuration(self, idx, loss):
        """process_evaluated_configuration(idx, loss)
        Determines if the configuration should be added to the population or not by calling `self.replace_worst_individual(idx, loss)`.

        Parameters
        ----------
        idx: int
            Index of the configuration.
        loss: float
            Loss of the evaluation.
        
        Returns
        --------
        delete: True
            Boolean indicating if the configuration extra-information should be deleted. In SSEA, only extra-information of the best model are kept.
        row_pop: dict
            Dictionary containing evaluation information to be saved within a `.csv` file called `computation_file.csv`.
        loss: float
            Loss of the configuration.
        """
        add = True
        if len(self.storage)>=self.population_size:
            add = self.replace_worst_individual(idx, loss)
        if add:
            self.storage[idx] =  {"Loss": loss}
        return True, pd.DataFrame.from_dict({"Idx": [idx], "Loss": [loss]}), loss

    def replace_worst_individual(self, idx, loss):
        """replace_worst_individualidx, loss)
        Determines if the configuration is better than the worst one from the population.
        If this is the case, removes the worst one from the population and returns True.
        Else, returns False.

        Parameters
        ----------
        idx: int
            Index of the configuration.
        loss: float
            Loss of the evaluation.
        
        Returns
        --------
        add: boolean
            Boolean indicating if the configuration should be added to the population.
        """
        idx_max_loss = list(self.storage.keys())[np.argmax([self.storage[i]['Loss'] for i in self.storage.keys()])]
        if loss < self.storage[idx_max_loss]['Loss']:
            self.storage.pop(idx_max_loss)
            logger.info(f'Replacing {idx_max_loss} by {idx}, removing {self.save_dir}/x_{idx_max_loss}.pkl')
            os.remove(f"{self.save_dir}/x_{idx_max_loss}.pkl")
            return True
        else:
            os.remove(f"{self.save_dir}/x_{idx}.pkl")
            logger.info(f'{idx} is the worst element, removing {self.save_dir}/x_{idx}.pkl.')
            return False
