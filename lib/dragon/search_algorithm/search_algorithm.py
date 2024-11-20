from datetime import datetime
import shutil
from abc import ABC, abstractmethod
import pickle
import os
import signal
from dragon.utils.tools import logger
import numpy as np
import pandas as pd


def set_mpi():
    try:
        from mpi4py import MPI
        mpi_dict = {
            "MPI": MPI,
            "comm": MPI.COMM_WORLD,
            "status": MPI.Status(),
        }
        mpi_dict["rank"] = mpi_dict['comm'].Get_rank()
        mpi_dict['p'] = mpi_dict['comm'].Get_size()
        return True, mpi_dict
    except Exception as e:
        logger.warning('Install mpi4py if you want to use the distributed version.')
        return False, None

@abstractmethod
class SearchAlgorithm(ABC):
    def __init__(self, search_space, n_iterations: int, init_population_size: int, evaluation, save_dir, models=None, pop_path = None, verbose=False):
        self.search_space = search_space
        self.n_iterations = n_iterations
        self.population_size = init_population_size
        self.evaluation = evaluation
        self.save_dir = save_dir
        mpi, mpi_dict = set_mpi()
        if mpi:
            self.run = self.run_mpi
            self.mpi_dict = mpi_dict
        else:
            self.run = self.run_no_mpi
        self.models = models
        self.storage = {}
        self.min_loss = np.inf
        self.pop_path = pop_path
        self.verbose=verbose     

    @abstractmethod
    def select_next_configurations(self, K):
        pass

    @abstractmethod
    def process_evaluated_configuration(self, **args):
        pass

    def process_evaluated_row(self, row):
        loss = row['Loss']
        self.storage[row['Idx']] = {"Loss": loss}
        if self.min_loss > loss:
            logger.info(f'Best found! {loss} < {self.min_loss}')
            self.min_loss = loss

    
    def create_population(self):            
        if self.models is None:
            self.models = []
        population = self.search_space.random(size=self.population_size-len(self.models)-len(self.storage))
        if len(self.models)>0:
            population += self.models
        for i,p in enumerate(population):
            with open(f"{self.save_dir}/x_{i}.pkl", "wb") as f: # savedir/x_i.pkl
                pickle.dump(p, f)
        logger.info(f'The whole population has been created (size = {len(population)}), {len(population) - len(self.models) - len(self.storage)} have been randomy initialized.')
    
    def recover_optimisation(self):
        logger.info(f"Generating population from {self.pop_path} containing {len(os.listdir(self.pop_path))} elements.")
        try:
            df_pop = pd.read_csv(self.pop_path+"/computation_file.csv")
            for _, row in df_pop.iterrows():
                self.process_evaluated_row(row)
            for i in list(self.storage.keys()):
                if not os.path.exists(f"{self.save_dir}/x_{i}.pkl"):
                    self.storage.pop(i)
        except FileNotFoundError:
            logger.error(f"{self.pop_path+'/computation_file.csv'} does not exist, starts from fresh population.")
            if os.path.exists(self.save_dir):
                shutil.rmtree(self.save_dir)
            os.makedirs(self.save_dir+"/best_model/")

    
    def evaluate_first_population_mpi(self):
        nb_send = 0
        # Dynamically send and evaluate the first population
        logger.info(f'We start by evaluating the whole population (size={self.population_size})')
        while (nb_send< self.population_size) and (nb_send < self.mpi_dict['p']-1):
            self.mpi_dict['comm'].send(dest=nb_send+1, tag=0, obj=(nb_send))
            nb_send +=1
        nb_receive = 0
        while nb_send < self.population_size:
            loss, idx = self.mpi_dict['comm'].recv(source=self.mpi_dict['MPI'].ANY_SOURCE, tag=0, status=self.mpi_dict['status'])
            source = self.mpi_dict['status'].Get_source()
            self.save_best_model(idx, loss)
            logger.info(f'Sending individual {nb_send} to processus {source}')
            self.mpi_dict['comm'].send(dest=source, tag=0, obj=(nb_send))
            nb_send+=1
            nb_receive +=1
        while nb_receive < self.population_size:
            loss, idx = self.mpi_dict['comm'].recv(
                source=self.mpi_dict['MPI'].ANY_SOURCE, tag=0, status=self.mpi_dict['status']
            )
            source = self.mpi_dict['status'].Get_source()
            self.save_best_model(idx, loss)
            nb_receive+=1

    def save_best_model(self, idx, loss):
        x_path = os.path.join(self.save_dir, f"{idx}")
        delete, row_pop, loss = self.process_evaluated_configuration(idx, loss)
        row_pop['TimeStamp'] = datetime.now()
        if np.isinf(loss):
            logger.info(f'Idx = {idx} has an infinite loss, deleting it.')
            if idx in self.storage.keys():
                self.storage.pop(idx)
            if os.path.exists(x_path) and os.path.isdir(x_path):
                shutil.rmtree(x_path)
        else:
            if loss < self.min_loss:
                logger.info(f'Best found! {loss} < {self.min_loss}')
                self.min_loss = loss
                try:
                    try:
                        shutil.copytree(x_path, self.save_dir+"/best_model/")
                    except FileExistsError:
                        shutil.rmtree(self.save_dir+"/best_model/")
                        shutil.copytree(x_path, self.save_dir+"/best_model/")
                except Exception:
                    if self.verbose:
                        logger.error(f'Failed to load x, idx= {idx}.', exc_info=True)
                    else:
                        logger.error(f'Failed to load x, idx= {idx}.')
            if delete:
                shutil.rmtree(x_path)
        try:
            df_pop = pd.read_csv(self.save_dir+"/computation_file.csv")
            df_pop = pd.concat((df_pop, row_pop), axis=0)
        except FileNotFoundError:
            df_pop = row_pop
        df_pop.to_csv(self.save_dir+"/computation_file.csv", index=False)
        
    def evaluate(self, idx, timed=False):
        with open(f"{self.save_dir}/x_{idx}.pkl", 'rb') as f:
            x = pickle.load(f)
        os.remove(f"{self.save_dir}/x_{idx}.pkl")
        x_path = f"{self.save_dir}/{idx}/"
        os.makedirs(x_path, exist_ok=True)
        try:
            if timed:
                loss = timed_evaluation(x, idx, 45*60, self.evaluation)
            else:
                loss = self.evaluation(x, idx)
            if not isinstance(loss, float):
                if len(loss) ==2:
                    loss, model = loss
                    if hasattr(model, "save"):
                        model.save(x_path)
                elif len(loss) == 3:
                    loss, model, x = loss
                    if hasattr(model, "save"):
                        model.save(x_path)
        except Exception as e:
            if self.verbose:
                logger.error(f'Worker with individual {idx} failed with {e}, set loss to inf', exc_info=True)
            else:
                logger.error(f'Worker with individual {idx} failed with {e}, set loss to inf')

            loss = np.inf
        with open(x_path + "/x.pkl", 'wb') as f: # savedir/idx/x.pkl
            pickle.dump(x, f)
        with open(f"{self.save_dir}/x_{idx}.pkl", "wb") as f: # savedir/x_idx.pkl
            pickle.dump(x, f)
        return loss, idx

    def run_mpi(self):
        rank = self.mpi_dict['rank']
        if rank == 0:
            logger.info(f"Master here ! starts search algorithm.")
            if self.pop_path is not None:
                self.recover_optimisation()
            else:
                if os.path.exists(self.save_dir):
                    logger.info(f'{self.save_dir} already exists. Deleting it.')
                    shutil.rmtree(self.save_dir)
                os.makedirs(self.save_dir+"/best_model/")

            if len(self.storage) < self.population_size:
                ### Create first population
                self.create_population()

                ### Evaluate first population
                self.evaluate_first_population_mpi()

            ### Start evolution
            self.K = len(self.storage)
            t = self.K

            # Store individuals waiting for a free processus
            to_evaluate = []
            nb_send = 0
            logger.info(f'After initialisation, it remains {self.n_iterations - t} iterations.')

            # Send first offspring to all processus
            while nb_send < self.mpi_dict['p']-1:
                if len(to_evaluate) == 0:
                    to_evaluate = self.select_next_configurations()
                idx = to_evaluate.pop()
                logger.info(f'Master, sending individual to processus {idx}')
                self.mpi_dict['comm'].send(dest=nb_send+1, tag=0, obj=(idx))
                del to_evaluate[idx]
                nb_send+=1
            
            # dynamically receive and send evaluations
            while t < (self.n_iterations-self.mpi_dict['p']-1):
                loss, idx = self.mpi_dict['comm'].recv(source=self.mpi_dict['MPI'].ANY_SOURCE, tag=0, status=self.mpi_dict['status'])
                t+=int(not np.isinf(loss))
                self.save_best_model(idx, loss)

                source = self.mpi_dict['status'].Get_source()
                if len(to_evaluate) == 0:
                    to_evaluate = self.select_next_configurations()    
                idx = to_evaluate.pop()
                logger.info(f'Master, sending individual to processus {idx}.')
                self.mpi_dict['comm'].send(dest=source, tag=0, obj=(idx))
                nb_send+=1
                
            nb_receive = 0
            # Receive last evaluation
            while (nb_receive < self.mpi_dict['p']-1):
                loss, idx = self.mpi_dict['comm'].recv(source=self.mpi_dict['MPI'].ANY_SOURCE, tag=0, status=self.mpi_dict['status'])
                nb_receive += int(not np.isinf(loss))
                source = self.mpi_dict['status'].Get_source()
                self.save_best_model(idx, loss)
                    
            logger.info(f"Search algorithm is done. Min Loss = {self.min_loss}")
            for i in range(1, self.mpi_dict['p']):
                self.mpi_dict['comm'].send(dest=i, tag=0, obj=(None))
            for x in os.listdir(self.save_dir):
                if os.path.isdir(self.save_dir+f"/{x}"):
                    if "best_model" not in x:
                        shutil.rmtree(self.save_dir+f"/{x}")
                else:
                    if ".csv" not in x:
                        os.remove(self.save_dir+f"/{x}")
        else:
            logger.info(f"Worker {rank} here.")
            stop = False
            while not stop:
                idx = self.mpi_dict['comm'].recv(source=0, tag=0, status=self.mpi_dict['status'])
                if idx is not None:
                    loss, idx = self.evaluate(idx, timed=True)
                    self.mpi_dict['comm'].send(dest=0, tag=0, obj=[loss, idx])
                else:
                    logger.info(f'Worker {rank} has been stopped')
                    stop = True

    def run_no_mpi(self):
        if self.pop_path is not None:
            self.recover_optimisation()
        else:
            if os.path.exists(self.save_dir):
                logger.info(f'{self.save_dir} already exists. Deleting it.')
                shutil.rmtree(self.save_dir)
            os.makedirs(self.save_dir+"/best_model/")        
        if len(self.storage) < self.population_size:
            ### Create first population
            self.create_population()

            ### Evaluate first population
            for idx in range(self.population_size):
                loss, idx = self.evaluate(idx)
                self.save_best_model(idx, loss)
            logger.info(f"All models have been at least evaluated once, t = {len(self.storage)} < {self.n_iterations}.")

        ### Start evolution
        self.K = len(self.storage)
        t = self.K
        logger.info(f'After initialisation, it remains {self.n_iterations - t} iterations.')
        to_evaluate = []
        while t < self.n_iterations:
            if len(to_evaluate) == 0:
                to_evaluate = self.select_next_configurations()    
            idx = to_evaluate.pop()
            loss, idx = self.evaluate(idx)
            t+=int(not np.isinf(loss))
            self.save_best_model(idx, loss)
        logger.info(f"Search algorithm is done. Min Loss = {self.min_loss}")
        for x in os.listdir(self.save_dir):
            if os.path.isdir(self.save_dir+f"/{x}"):
                if "best_model" not in x:
                    shutil.rmtree(self.save_dir+f"/{x}")
            else:
                if ".csv" not in x:
                    os.remove(self.save_dir+f"/{x}")

def timed_evaluation(x, idx, max_duration, evaluation):
    def handler(signum, frame):
        print(f'Evaluation of model {idx} took more than {max_duration} seconds. Stopping evaluation.')
        raise TimeoutError()
    signal.signal(signal.SIGALRM, handler)
    try:
        signal.alarm(max_duration)
        result = evaluation(x, idx)
    except TimeoutError:
        result = np.inf, None 
    finally:
        signal.alarm(0)
    return result