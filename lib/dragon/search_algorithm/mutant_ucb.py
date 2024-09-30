from copy import deepcopy
import os
import pickle
import shutil
import numpy as np
import pandas as pd
import torch
from dragon.utils.tools import logger

def save_best_model(storage, min_loss, save_dir, idx):
    model = storage[idx]
    loss = model['Loss']
    if loss < min_loss:
        logger.info(f'Best found! {loss} < {min_loss}')
        min_loss = loss
        x_path = model["Individual"]
        if not isinstance(x_path, str):
            x = x_path
            x_path = os.path.join(save_dir, f"{idx}")
            with open(x_path + "/x.pkl", 'wb') as f:
                    pickle.dump(x, f)
        try:
            if len(os.listdir(x_path))>1:
                try:
                    shutil.copytree(x_path, save_dir+"/best_model/")
                except FileExistsError:
                    shutil.rmtree(save_dir+"/best_model/")
                    shutil.copytree(x_path, save_dir+"/best_model/")
        except torch.cuda.OutOfMemoryError:
            logger.error(f'Failed to load x, idx= {idx}.')
    return min_loss

class Mutant_UCB:
    def __init__(self, search_space, save_dir, T, N, K, E, evaluation, **args):
        self.T = T
        self.N = N
        self.K = K
        self.E = E
        self.search_space = search_space
        try:
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD
            self.status = MPI.Status()
            self.p_name = MPI.Get_processor_name()
            self.rank = self.comm.Get_rank()
            self.p = self.comm.Get_size()
            self.run = self.run_mpi
        except Exception as e:
            logger.warning('Install mpi4py if you want to use the distributed version.')
            self.run = self.run_no_mpi
        self.evaluation = evaluation
        self.save_dir = save_dir

    def run_no_mpi(self):
        min_loss = np.inf
        population = self.search_space.random(self.K)
        # First round
        storage = {}
        for idx, x in enumerate(population):
            x_path = os.path.join(self.save_dir, f"{idx}")
            loss = self.evaluation(x, idx=idx)
            os.makedirs(x_path, exist_ok=True)
            if not isinstance(loss, float):
                if len(loss) ==2:
                    loss, model = loss
                    if hasattr(model, "save"):
                        model.save(x_path)
                elif len(loss) == 3:
                    loss, model, x = loss
                    if hasattr(model, "save"):
                        model.save(x_path)
            with open(x_path + "/x.pkl", 'wb') as f:
                pickle.dump(x, f)
                del x
            storage[idx]= {"Individual": x_path, "N": 1, "N_bar":1, "UCBLoss": loss, "Loss": loss}
        t = self.K
        sent = {}
        while t < self.T:
            storage, sent, idx = self.ucb_iteration(storage, sent)
            x = sent[idx]['Individual']
            if isinstance(x, str):
                with open(x+"/x.pkl", 'rb') as f:
                    x = pickle.load(f)
                    os.remove(x_path+"/x.pkl")
            loss = self.evaluation(x, idx=idx)
            os.makedirs(x_path, exist_ok=True)
            if not isinstance(loss, float):
                if len(loss) ==2:
                    loss, model = loss
                    if hasattr(model, "save"):
                        model.save(x_path)
                elif len(loss) == 3:
                    loss, model, x = loss
                    if hasattr(model, "save"):
                        model.save(x_path)
            with open(x_path + "/x.pkl", 'wb') as f:
                pickle.dump(x, f)
                del x
            sent[idx]['Individual'] = x_path
            sent[idx]['Loss'] = loss
            sent[idx]['UCBLoss'] = (loss + sent[idx]['N_bar']*sent[idx]['UCBLoss'])/(sent[idx]['N_bar']+1)
            sent[idx]['N'] +=1
            sent[idx]['N_bar'] +=1
            storage[idx] = sent.pop(idx)
            min_loss = save_best_model(storage, min_loss, self.save_dir, idx)
            t+=1
        logger.info(f"Mutant-UCB is done. Min Loss = {min_loss}")
        return min_loss

    def run_initialization(self):
        from mpi4py import MPI
        rank = self.comm.Get_rank()  
        min_loss = np.inf
        storage = {}
        population = self.search_space.random(self.K)
        logger.info(f'The whole population has been created (size = {len(population)})')
        nb_send = 0
        t = 0
        # Dynamically send and evaluate the first population
        while (nb_send< len(population)) and (nb_send < self.p-1):
            x = population[nb_send]
            logger.info(f'Master sends individual {nb_send} to processus {nb_send+1} < {self.p}')
            self.comm.send(dest=nb_send+1, tag=0, obj=(x, nb_send))
            nb_send +=1
            t+=1
        nb_receive = 0
        while nb_send < self.K:
            loss, x_path, idx = self.comm.recv(
                source=MPI.ANY_SOURCE, tag=0, status=self.status
            )
            source = self.status.Get_source()
            storage[idx] = {"Individual": x_path, "N": 1, "N_bar": 1, "UCBLoss": loss, "Loss": loss}
            min_loss = save_best_model(storage, min_loss, self.save_dir, idx)
            x = population[nb_send]
            logger.info(f'Master sends individual {nb_send} to processus {source}')
            self.comm.send(dest=source, tag=0, obj=(x, nb_send))
            nb_send+=1
            nb_receive +=1

        while nb_receive < self.K:
            loss, x_path, idx = self.comm.recv(
                source=MPI.ANY_SOURCE, tag=0, status=self.status
            )
            source = self.status.Get_source()
            storage[idx] = {"Individual": x_path, "N": 1, "N_bar": 1, "UCBLoss": loss, "Loss": loss}
            min_loss = save_best_model(storage, min_loss, self.save_dir, idx)   
            nb_receive+=1
            t+=1
        logger.info(f"Initialization is done: all models have been at least evaluated once.")
        return storage, min_loss

    def run_mpi(self, rs_pop=None):
        from mpi4py import MPI
        rank = self.comm.Get_rank()  
        if rank ==0:
            os.makedirs(self.save_dir, exist_ok=True)
            save_file = self.save_dir+"/best_model.csv"
            logger.info(f"Master here ! start UCB algorithm.")
            if rs_pop is None:
                storage, min_loss = self.run_initialization()
            else:
                storage, min_loss = generate_rs_pop(rs_pop)
            t = len(storage)
            logger.info(f"After initialization, it remains {self.T-t} iterations.")
            nb_send = 0
            # dynamically receive and send evaluations
            sent = {}
            while nb_send < self.p-1:
                sent_bool = False
                while not sent_bool:
                    # perform ucb
                    storage, sent, idx = self.ucb_iteration(storage, sent)
                    logger.info(f'Master sends individual {idx} to processus {nb_send+1}')
                    try:
                        self.comm.send(dest=nb_send+1, tag=0, obj=(sent[idx]['Individual'], idx))
                        sent_bool = True
                    except OverflowError:
                        logger.error(f"Master {rank}, failed to sent {idx} to {nb_send}. Removing {idx} from pool.")
                        sent.pop(idx)                    
                nb_send +=1
                t+=1
            while len(storage)>0 and t < self.T - self.N + 1:
                loss, x_path, idx = self.comm.recv(
                    source=MPI.ANY_SOURCE, tag=0, status=self.status
                )
                source = self.status.Get_source()
                sent[idx]["Individual"] =  x_path
                sent[idx]['Loss'] = loss
                sent[idx]['UCBLoss'] = (loss + sent[idx]['N_bar']*sent[idx]['UCBLoss'])/(sent[idx]['N_bar']+1)
                sent[idx]['N'] +=1
                sent[idx]['N_bar'] +=1
                storage[idx] = sent.pop(idx)
                min_loss = save_best_model(storage, min_loss, self.save_dir, idx)

                sent_bool = False
                while not sent_bool:
                    if len(storage) == 0:
                        logger.info(f'Storage is empty.')
                        sent_bool = True
                        break
                    storage, sent, idx = self.ucb_iteration(storage, sent)
                    logger.info(f'Master sends individual {idx} to processus {source}')
                    try:
                        self.comm.send(dest=source, tag=0, obj=(sent[idx]['Individual'], idx))
                        sent_bool = True
                    except OverflowError:
                        logger.error(f"Failed to sent {idx} to {source}. Removing {idx} from pool.")
                        sent.pop(idx) 
                nb_send +=1
                t+=1
            
            nb_receive = 0
            # Receive last evaluation
            while nb_receive < self.p-1:
                loss, x_path, idx = self.comm.recv(
                    source=MPI.ANY_SOURCE, tag=0, status=self.status
                )
                source = self.status.Get_source()
                logger.info(f"Master {rank}, last round for processus number {source}")
                sent[idx]["Individual"] =  x_path
                sent[idx]['Loss'] = loss
                sent[idx]['UCBLoss'] = (loss + sent[idx]['N_bar']*sent[idx]['UCBLoss'])/(sent[idx]['N_bar']+1)
                sent[idx]['N'] +=1
                sent[idx]['N_bar'] +=1
                storage[idx] = sent.pop(idx)
                min_loss = save_best_model(storage, min_loss, self.save_dir, idx)
                nb_receive+=1            
            
            logger.info(f"Mutant-UCB is done. Min loss = {min_loss}")
            for i in range(1, self.p):
                self.comm.send(dest=i, tag=0, obj=(None,None))
            return min_loss
        else:
            logger.info(f"Worker {rank} here.")
            stop = True
            while stop:
                x, idx = self.comm.recv(source=0, tag=0, status=self.status)
                if idx != None:
                    x_path = os.path.join(self.save_dir, f"{idx}")
                    if isinstance(x, str):
                        with open(x+"/x.pkl", 'rb') as f:
                            x = pickle.load(f)
                            os.remove(x_path+"/x.pkl")
                    loss = self.evaluation(x, idx=idx)
                    os.makedirs(x_path, exist_ok=True)
                    if not isinstance(loss, float):
                        if len(loss) ==2:
                            loss, model = loss
                            if hasattr(model, "save"):
                                model.save(x_path)
                        elif len(loss) == 3:
                            loss, model, x = loss
                            if hasattr(model, "save"):
                                model.save(x_path)

                    with open(x_path + "/x.pkl", 'wb') as f:
                        pickle.dump(x, f)
                    self.comm.send(dest=0, tag=0, obj=[loss, x_path, idx])
                else:
                    logger.info(f'Worker {rank} has been stopped')
                    stop = False

    def ucb_iteration(self, storage, sent):
        # Compute ucb loss
        ucb_losses = [storage[i]['UCBLoss'] - np.sqrt(self.E/storage[i]['N']) for i in storage.keys()]
        idx = list(storage.keys())[np.argmin(ucb_losses)]
        # Mutation probability
        mutation_p = storage[idx]['N_bar'] / self.N
        # Random variable
        r = np.random.binomial(1, mutation_p, 1)[0]
        if r == 0:
            # Keep Training, remove the model from the storage
            logger.info(f'With p = {mutation_p} = {storage[idx]["N_bar"]} / {self.N}, training {idx} instead')
            sent[idx] = storage.pop(idx) 
        else:
            # Mutate the model
            logger.info(f'With p = {mutation_p} = {storage[idx]["N_bar"]} / {self.N}, mutating {idx} to {self.K}')
            storage[idx]['N'] +=1
            # Load model
            if isinstance(storage[idx]['Individual'], str):
                with open(storage[idx]['Individual']+"/x.pkl", 'rb') as f:
                    old_x = pickle.load(f)
            else:
                old_x = storage[idx]['Individual']
            # mutate the model
            not_muted = True
            while not_muted:
                try:
                    x = self.search_space.neighbor(deepcopy(old_x))
                    not_muted = False
                except Exception as e:
                    logger.error(f"While mutating, an exception was raised: {e}")
                    logger.error(f'Old x is: {old_x}')
            idx = self.K
            self.K += 1
            sent[idx] = {"Individual": x, "N": 0, "N_bar": 0, "UCBLoss": 0}
        return storage, sent, idx

def generate_rs_pop(rs_pop):
        logger.info(f"Generating population from {rs_pop}\n{os.listdir(rs_pop)}")
        storage = {}
        min_loss = np.inf
        for dir_name in os.listdir(rs_pop):
            try:
                idx = int(dir_name)
                loss = float(pd.read_csv(rs_pop + "/" + dir_name + "/" + "best_model_archi.csv", sep=";").iloc[-1,1])
                storage[idx] = {"Individual": rs_pop + "/" + dir_name, "N": 1, "N_bar": 1, "UCBLoss": loss, "Loss": loss}
                logger.info(f"Adding {idx} to storage with loss = {loss}")
                if loss < min_loss:
                    min_loss = loss
            except Exception as e:
                logger.error(f"Error when recovering rs pop: {e}", exc_info=True)
        return storage, min_loss