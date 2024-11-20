from copy import deepcopy
import os
import pickle
from dragon.search_algorithm.search_algorithm import SearchAlgorithm
import numpy as np
import pandas as pd
from dragon.utils.tools import logger

class Mutant_UCB(SearchAlgorithm):
    def __init__(self, search_space, save_dir, T, N, K, E, evaluation, models=None, pop_path=None, verbose=False, **args):
        super(Mutant_UCB, self).__init__(search_space=search_space, 
                                            n_iterations=T, 
                                            init_population_size=K, 
                                            evaluation=evaluation, 
                                            save_dir=save_dir, 
                                            models=models, pop_path=pop_path, 
                                            verbose=verbose)
        
    
        self.N = N
        self.E = E
        self.sent = {}


    def select_next_configurations(self):
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
    
    def process_evaluated_row(self, row):
        loss = row['UCBLoss']
        self.storage[row['Idx']] = {"Loss": row['Loss'], "N": row['N'], "N_bar": row['N_bar'], "UCBLoss": loss}
        if self.min_loss > loss:
            logger.info(f'Best found! {loss} < {self.min_loss}')
            self.min_loss = loss

    def process_evaluated_configuration(self, idx, loss):
        if idx in self.sent.keys():
            self.sent[idx]['Loss'] = loss
            self.sent[idx]['UCBLoss'] = (loss + self.sent[idx]['N_bar']*self.sent[idx]['UCBLoss'])/(self.sent[idx]['N_bar']+1)
            self.sent[idx]['N'] +=1
            self.sent[idx]['N_bar'] +=1
            self.storage[idx] = self.sent.pop(idx)
        else:
            self.storage[idx] = {"N": 1, "N_bar": 1, "UCBLoss": loss, "Loss": loss}
        return False, pd.DataFrame({k: [v] for k, v in self.storage[idx].items()}), loss

