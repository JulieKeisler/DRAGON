from copy import deepcopy
import os
import pickle
from dragon.search_algorithm.search_algorithm import SearchAlgorithm
import numpy as np
import pandas as pd
from dragon.utils.tools import logger

class Mutant_UCB(SearchAlgorithm):
    """Mutant_UCB

    Search algorithm implementing the Mutant-UCB search algorithm inheriting from the `SearchAlgorithm` class.
    It implements a `select_next_configuration` and a `process_evaluated_configuration` methods, specific to Mutant-UCB.

    Parameters
    ----------
    search_space: `Variable`
        `Variable` containing all the design choices from the search space. It should implement a `random` method and a `neighbor` one.
    T: int
        Number of iterations.
    K: int
        Size of the population.
    N: int
        Maximum number of partial training for one configuration.
    E: float
        Exploratory parameters.
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
    time_max: int, default=45
        Maximum number of time (in minutes) for one evaluation.

    Attributes
    ----------
    N: int
        Maximum number of partial training for one configuration.
    E: float
        Exploratory parameters.
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
    time_max: int, default=45
        Maximum number of time (in minutes) for one evaluation.

    Example
    --------
    >>> from dragon.search_space.base_variables import ArrayVar
    >>> from dragon.search_operators.base_neighborhoods import ArrayInterval
    >>> from dragon.search_algorithm.mutant_ucb import Mutant_UCB
    >>> search_space = ArrayVar(dag, label="Search Space", neighbor=ArrayInterval())
    >>> search_algorithm = Mutant_UCB(search_space, save_dir="save/test_mutant", T=20, N=5, K=5, E=0.01, evaluation=loss_function)
    >>> search_algorithm.run()
    """
    def __init__(self, search_space, T, K, N, E, evaluation, save_dir, models=None, pop_path=None, verbose=False, **args):
        super(Mutant_UCB, self).__init__(search_space=search_space, 
                                            n_iterations=T, 
                                            init_population_size=K, 
                                            evaluation=evaluation, 
                                            save_dir=save_dir, 
                                            models=models, pop_path=pop_path, 
                                            verbose=verbose,
                                            time_max=45)
        
    
        self.N = N
        self.E = E
        self.sent = {}


    def select_next_configurations(self):
        """select_next_configurations()
        Defines a selection strategy for Mutant-UCB.
        Select the next configuration optimistically: the minimum loss + UCB interval.
        With a certain probability remove the configuration from `self.storage` and add it to `self.sent` to perform a new partial training.
        If not, increments the number of time the configuration has been picked, creates a new configuration using the `neighbor` attribute, increments the number of models within the population `self.K` and add the configuration to `self.sent`.
        
        Returns
        ----------
        [idx]: list
            List containing the idx of the selected configuration.
        """
        # Compute ucb loss
        iterated = False
        while not iterated:
            tries = 0
            ucb_losses = [self.storage[i]['UCBLoss'] - np.sqrt(self.E/self.storage[i]['N']) for i in self.storage.keys()]
            idx = list(self.storage.keys())[np.argmin(ucb_losses)]
            try:
                # Mutation probability
                mutation_p = self.storage[idx]['N_bar'] / self.N
                # Random variable
                r = np.random.binomial(1, mutation_p, 1)[0]
                if r == 0:
                    # Keep Training, remove the model from the storage
                    logger.info(f'With p = {mutation_p} = {self.storage[idx]["N_bar"]} / {self.N}, training {idx} instead')
                    self.sent[idx] = self.storage.pop(idx) 
                else:
                    # Mutate the model
                    logger.info(f'With p = {mutation_p} = {self.storage[idx]["N_bar"]} / {self.N}, mutating {idx} to {self.K}')
                    self.storage[idx]['N'] +=1
                    # Load model
                    with open(f"{self.save_dir}/x_{idx}.pkl", 'rb') as f:
                        old_x = pickle.load(f)
                    # mutate the model
                    x = self.search_space.neighbor(deepcopy(old_x))
                    idx = self.K
                    self.K+=1
                    with open(f"{self.save_dir}/x_{idx}.pkl", 'wb') as f:
                        pickle.dump(x, f)
                    del x
                    self.sent[idx] = {"N": 0, "N_bar": 0, "UCBLoss": 0}
                iterated = True
            except Exception as e:
                tries +=1
                if tries < 5:
                    logger.error(f"While ucb iration, an exception was raised: {e}, attempt {tries}/5.")
                else:
                    self.storage.pop(idx)
                    if os.path.exists(f"{self.save_dir}/x_{idx}.pkl"):
                        os.remove(f"{self.save_dir}/x_{idx}.pkl")
                    logger.error(f"While ucb iration, an exception was raised: {e}, removing {idx} from population. Size storage: {len(self.storage)}.")
        return [idx]
    
    def process_evaluated_configuration(self, idx, loss):
        """process_evaluated_configuration(idx, loss)
        Defines how to process the last evaluated configuration given its loss.
        Save the current loss, average loss according to previous evaluations.
        Increments the number of time the configuration has been picked and evaluated.
        Removes the configuration from `self.sent` and add it to `self.storage`.

        Parameters
        ----------
        idx: int
            Index of the configuration.
        loss: float
            Loss of the evaluation.
        
        Returns
        --------
        delete: False
            Boolean indicating if the configuration extra-information should be deleted. In Mutant-UCB, all evaluated configurations are kept.
        row_pop: dict
            Dictionary containing evaluation information to be saved within a `.csv` file called `computation_file.csv`.
        loss: float
            Average loss across all evaluations.
        """
        if idx in self.sent.keys():
            self.sent[idx]['Loss'] = loss
            self.sent[idx]['UCBLoss'] = (loss + self.sent[idx]['N_bar']*self.sent[idx]['UCBLoss'])/(self.sent[idx]['N_bar']+1)
            self.sent[idx]['N'] +=1
            self.sent[idx]['N_bar'] +=1
            self.storage[idx] = self.sent.pop(idx)
        else:
            self.storage[idx] = {"N": 1, "N_bar": 1, "UCBLoss": loss, "Loss": loss}
        return False, pd.DataFrame({k: [v] for k, v in self.storage[idx].items()}), loss
    
    def process_evaluated_row(self, row):
        """process_evaluated_row(row)
        Modifies the `process_evaluated_row` method from `SearchAlgorithm` to add the extra information contained by `computation_file.csv`.
        Add the average loss, the number of times the configuration has been picked and evaluated to the `self.storage` dictionary.

        Parameters
        ----------
        row: dict
            Dictionary containing the information of an evaluated configuration.
        """
        loss = row['UCBLoss']
        self.storage[row['Idx']] = {"Loss": row['Loss'], "N": row['N'], "N_bar": row['N_bar'], "UCBLoss": loss}
        if self.min_loss > loss:
            logger.info(f'Best found! {loss} < {self.min_loss}')
            self.min_loss = loss

 

