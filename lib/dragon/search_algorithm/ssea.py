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
    def __init__(self, search_space, n_iterations: int, population_size: int, selection_size: int, evaluation, save_dir, models = None, pop_path=None, crossover=DAGTwoPoint(), verbose=False, **args):
        super(SteadyStateEA, self).__init__(search_space=search_space, 
                                            n_iterations=n_iterations, 
                                            init_population_size=population_size, evaluation=evaluation, save_dir=save_dir, models=models, pop_path=pop_path, verbose=verbose)
        self.selection_size = selection_size
        self.crossover = crossover
        

    def select_next_configurations(self):
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
                self.storage[best_1] = parent1
        not_selected = True
        while not_selected:
            best_2 = selection[np.argmin([self.storage[i]['Loss'] for i in selection])]
            try:
                parent2 = self.storage[best_2]
                self.storage[best_2] = parent2
                with open(f"{self.save_dir}/x_{best_2}.pkl", 'rb') as f:
                    x2 = pickle.load(f)
                not_selected = False
            except Exception as e:
                logger.error(f'Could not load individual {best_2}/{len(list(self.storage.keys()))}, {e}')
        self.storage[best_1] = parent1
        offspring_1, offspring_2 = deepcopy(x1), deepcopy(x2)
        self.crossover(offspring_1, offspring_2)
        not_muted = True
        while not_muted:
            try:
                offspring_1 = self.search_space.neighbor(deepcopy(offspring_1))
                not_muted = False
            except Exception as e:
                logger.error(f"While mutating, an exception was raised: {e}")
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
        add = True
        if len(self.storage)>=self.population_size:
            add = self.replace_worst_individual(idx, loss)
        if add:
            self.storage[idx] =  {"Loss": loss}
        return True, pd.DataFrame.from_dict({"Idx": [idx], "Loss": [loss]}), loss

    def replace_worst_individual(self, idx, loss):
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