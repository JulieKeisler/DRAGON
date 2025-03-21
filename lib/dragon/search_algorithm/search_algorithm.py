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
    """set_mpi()
        Verifies if the distributed version can be used.
        Try to import the package `mpi4py` and verifies that at least two processes are available.
    
    Returns
    --------
        mpi_bool: boolean
            Indicating if the distributed version can be used
        mpi_dict: dict
            Dictionary containing the MPI information.
        """
    try:
        from mpi4py import MPI
        mpi_dict = {
            "MPI": MPI,
            "comm": MPI.COMM_WORLD,
            "status": MPI.Status(),
        }
        mpi_dict["rank"] = mpi_dict['comm'].Get_rank()
        mpi_dict['p'] = mpi_dict['comm'].Get_size()
        if mpi_dict['p'] == 1:
            logger.warning('Use multiple processes if you want to use the distributed version.')
            return False, None
        else:
            return True, mpi_dict
    except Exception as e:
        logger.warning('Install mpi4py if you want to use the distributed version.')
        return False, None

@abstractmethod
class SearchAlgorithm(ABC):
    """SearchAlgorithm

    Abstract class describing the general structure of a search algorithm.
    The classes inheriting from the `SearchAlgorithm` abstract class should implement a `select_next_configuration` and a `process_evaluated_configuration` methods.

    Parameters
    ----------
    search_space: `Variable`
        `Variable` containing all the design choices from the search space. It should implement a `random` method and a `neighbor` one if necessary.
    n_iterations: int
        Number of iterations.
    init_population_size: int
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
    time_max: int, default=45
        Maximum number of time (in minutes) for one evaluation.

    Attributes
    ----------
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

    """
    def __init__(self, search_space, n_iterations: int, init_population_size: int, evaluation, save_dir, models=None, pop_path = None, verbose=False, time_max=45):
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
        self.time_max=time_max

    @abstractmethod
    def select_next_configurations(self):
        """select_next_configurations()
        Defines a selection strategy for the current search algorithm potentially based on the current population and the previous evaluations.
        It should save, if needed, the next configurations to evaluate within `save_dir` and returns their indexes as a list.
        This function depends on the type of search algorithm.
        """
        pass

    @abstractmethod
    def process_evaluated_configuration(self, idx, loss):
        """process_evaluated_configuration(idx, loss)
        Defines how to process the last evaluated configuration given its loss.
        This function depends on the type of search algorithm.

        Parameters
        ----------
        idx: int
            Index of the configuration.
        loss: float
            Loss of the evaluation.
        
        Returns
        --------
        delete: boolean
            Boolean indicating if the configuration extra-information should be deleted.
        row_pop: dict
            Dictionary containing evaluation information to be saved within a `.csv` file called `computation_file.csv`.
        loss: float
            Loss of the evaluation according to the algorithm (might be different than the one returned by the performance evaluation function).
        """
        pass

  
    def create_population(self):
        """create_population()
        Create initial population.
        Integrated the configurations contained within `self.models`, randomly initialize the others.
        The configuration are saved within `{self.save_dir}/x_{i}.pkl`, where `i` is the configuration unique idx.

        """
  
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
        """recover_population()
        Recovers an already initialized population.
        The basis is a `.csv` path generated by the algorithm called `computation_file.csv` saved within `self.pop_path`.
        This file contains the information of already evaluated configuration.
        Each row corresponds to a unique configuration.
        The function process each row to fill the `self.storage` dictionary.
        If the `.csv` is not found, the content of `self.pop_path` is removed and the population starts from a fresh population.
        """
        logger.info(f"Generating population from {self.pop_path} containing {len(os.listdir(self.pop_path))} elements.")
        try:
            df_pop = pd.read_csv(self.pop_path+"/computation_file.csv")
            for _, row in df_pop.iterrows():
                self.process_evaluated_row(row)
            for i in list(self.storage.keys()):
                if not os.path.exists(f"{self.save_dir}/x_{i}.pkl"):
                    self.storage.pop(i)
            logger.info(f"Recovered population has size of {len(self.storage)}.")
        except FileNotFoundError:
            logger.error(f"{self.pop_path+'/computation_file.csv'} does not exist, starts from fresh population.")
            if os.path.exists(self.save_dir):
                shutil.rmtree(self.save_dir)
            os.makedirs(self.save_dir+"/best_model/")

    def process_evaluated_row(self, row):
        """process_evaluated_row(row)
        Fill the `self.storage` dictionary with the configuration evaluation information stored in `row`.
        This function is called by `self.recover_optimisation`.

        Parameters
        ----------
        row: dict
            Dictionary containing the information of an evaluated configuration.
        """

        loss = row['Loss']
        self.storage[row['Idx']] = {"Loss": loss}
        if self.min_loss > loss:
            logger.info(f'Best found! {loss} < {self.min_loss}')
            self.min_loss = loss

    def save_best_model(self, idx, loss):
        """save_best_model(idx, loss)
        Process the evaluated configuration by calling the `self.process_evaluated_configuration` method.
        If the `loss` is infinite, remove the model from the population.
        Else, verify if this is the best model found so far. 
        If it is, save its information into the `self.save_dir+"/best_model/"` directory.
        Add the evaluation information into `computation_file.csv`.

        Parameters
        ----------
        idx: int
            Index of the configuration.
        loss: float
            Loss of the evaluation.
        """
        
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
    
  
    def evaluate_first_population_mpi(self):
        """evaluate_first_population_mpi()
        Distributed evaluation (with MPI) of an initialized population.
        Iteratively sends the idx in range from 0 to `self.population_size` to the workers.
        When an evaluation comes back, call the `self.save_best_model` method to save it within the `self.storage` dictionary, 

        """
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

    
    def evaluate(self, idx):
        """evaluate()
        Evaluation function of the configuration `idx`.
        Load the corresponding configuration from `{self.save_dir}/x_{idx}.pkl`.
        Called the `self.evaluation` function.
        If the model is returned, called the `save` method from this model to save its extra information (prediction, weights, etc) into `{self.save_dir}/{idx}/`.
        Store the configuration modified (or not) by the evaluation function back to `{self.save_dir}/x_{idx}.pkl` and `{self.save_dir}/{idx}/`.
        The path `{self.save_dir}/{idx}/` will be deleted by the lagorithm if the model is not the best one from the population.

        Parameters
        -----------
        idx: int
            Index of the configuration to evaluated.
        Returns
        --------
        loss: float
            Loss found by the performance evaluation function.
        idx: int
            Index of the configuration.

        """
         
        with open(f"{self.save_dir}/x_{idx}.pkl", 'rb') as f:
            x = pickle.load(f)
        os.remove(f"{self.save_dir}/x_{idx}.pkl")
        x_path = f"{self.save_dir}/{idx}/"
        os.makedirs(x_path, exist_ok=True)
        try:
            if self.time_max is not None:
                loss = timed_evaluation(x, idx, self.time_max*60, self.evaluation)
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
        """run_mpi()
        Distributed version (with MPI) of the main function performing the search algorithm run.

        If the current process is the master process:
            - Recovers the previous optimisation run if `self.pop_path` is not None, else, delete `self.save_dir` if the path already exists.
            - Creates and evaluates the initial population using `self.create_population()` and `self.evaluate_first_population_mpi()`.
            - While the maximum number of iterations has not been reached:
                - Selects the next configurations to evaluated.
                - Dynamically sends them to the workers.
                - When receiving an evaluated configuration, processes it with the `self.save_best_model` method
                - At the end, removes all files produced by the search algorithm except the `self.save_dir+"/best_model/"` directory and `computation_file.csv`.
        Else:
            - Evaluates the configuration sent by the master process (using `self.evaluate`).
            - Sends back the configuration and its loss to the master.

        """
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

            if (self.population_size - len(self.storage)) < self.mpi_dict['p']:
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
                nb_receive += 1
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
                    loss, idx = self.evaluate(idx)
                    self.mpi_dict['comm'].send(dest=0, tag=0, obj=[loss, idx])
                else:
                    logger.info(f'Worker {rank} has been stopped')
                    stop = True

    def run_no_mpi(self):
        """run_no_mpi()
        Main function (without distribution) performing the search algorithm run.

        - Recovers the previous optimisation run if `self.pop_path` is not None, else, delete `self.save_dir` if the path already exists.
        - Creates and evaluates the initial population.
        - While the maximum number of iterations has not been reached:
            - Selects the next configurations to evaluated.
            - Evaluates them using the `self.evaluate` function.
            - Processes it with the `self.save_best_model` method
            - At the end, removes all files produced by the search algorithm except the `self.save_dir+"/best_model/"` directory and `computation_file.csv`.

        """
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
