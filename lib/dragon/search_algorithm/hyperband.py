
import os
import pickle
import shutil
import numpy as np
import torch

from dragon_old.utils.tools import logger


class HyperBand:
    def __init__(
            self,
            search_space,
            trainer,
            T, # Budget
            R, # Amount of ressources
            eta,
            config,
            lr,
            checkpoint,
            **kwargs
    ):
        from mpi4py import MPI #type: ignore
        self.search_space = search_space
        self.trainer=trainer
        self.T = T # Total budget
        self.eta = eta
        self.R = R # Amount of ressources
        self.Smax = int(np.floor(np.emath.logn(eta, R))) # Nombre d'itération
        self.B = T /(self.Smax + 1)
        self.comm = MPI.COMM_WORLD
        self.status = MPI.Status()
        self.p_name = MPI.Get_processor_name()
        self.rank = self.comm.Get_rank()
        self.p = self.comm.Get_size()
        self.config = config
        self.lr = lr
        self.checkpoint = checkpoint

    def run(self):
        if self.rank == 0:
            logger.info(f"HyperBand here! T = {self.T}, B={self.B}, R={self.R}, Smax={self.Smax}, eta={self.eta}")
            best_model = None
            min_loss = np.inf
            nb = 0
            for s in range(self.Smax+1, 0, -1):
                n = int(np.ceil((self.B / self.R) * (self.eta**s/(s+1)))) # number of config envaluated in this loop
                r = self.R * self.eta**(-s) # min ressources allocated to each configuration
                pop = self.search_space.random(n)
                logger.info(f'The population {s} has been created (size = {len(pop)})')
                for i in range(0, s+1):
                    n_i = int(np.floor(n * self.eta**(-i)))
                    r_i = int(np.ceil(r * self.eta**i))
                    logger.info(f"Pop size: {len(pop)}, number of epochs: {r_i} x {self.trainer.config['LoopSize']}")
                    self.trainer.config['NumEpochs'] = r_i*self.trainer.config['LoopSize']
                    nb_send = 0
                    nb_receive = 0
                    while nb_send < len(pop) and nb_send < self.p-1:
                        x = pop[nb_send]
                        if isinstance(x, dict):
                            idx = x['Idx']
                            x = x['Individual']
                        else:
                            idx=nb
                            nb+=1
                        logger.info(f'Sending individual {idx} to processus {nb_send+1} < {self.p}')
                        self.comm.send(dest=nb_send+1, tag=0, obj=("evaluate", x, self.trainer.config['NumEpochs'], idx))
                        nb_send +=1
                    population = []
                    while nb_send < len(pop):
                        (current_loss, loss), x_path, idx = self.comm.recv(source=MPI.ANY_SOURCE, tag=0, status=self.status)
                        source = self.status.Get_source()
                        shutil.move(f"{x_path}/x.pkl", f"{self.config['SaveDir']}/x_{idx}.pkl")
                        population.append({"Idx": idx, "Individual": f"{self.config['SaveDir']}/x_{idx}.pkl", "Loss": loss})
                        if loss < min_loss:
                            logger.info(f"Best found, idx={idx} ! {loss} < {min_loss}")
                            min_loss = loss
                            best_model = x
                            if len(os.listdir(x_path))>1:
                                try:
                                    shutil.copytree(x_path, self.config["SaveDir"]+"/best_model/")
                                except FileExistsError:
                                    shutil.rmtree(self.config["SaveDir"]+"/best_model/")
                                    shutil.copytree(x_path, self.config["SaveDir"]+"/best_model/")
                        shutil.rmtree(x_path) #type: ignore
                        x = pop[nb_send]
                        if isinstance(x, dict):
                            idx = x['Idx']
                            x = x['Individual']
                        else:
                            idx=nb
                            nb+=1
                        logger.info(f'Sending individual {idx} to processus {source}')
                        self.comm.send(dest=source, tag=0, obj=("evaluate", x, self.trainer.config['NumEpochs'], idx))
                        nb_send+=1
                        nb_receive +=1
                    while nb_receive < len(population):
                        (current_loss, loss), x_path, idx = self.comm.recv(
                            source=MPI.ANY_SOURCE, tag=0, status=self.status
                        )
                        source = self.status.Get_source()
                        shutil.move(f"{x_path}/x.pkl", f"{self.config['SaveDir']}/x_{idx}.pkl")
                        population.append({"Idx": idx, "Individual": f"{self.config['SaveDir']}/x_{idx}.pkl", "Loss": loss})
                        if loss < min_loss:
                            logger.info(f"Best found, idx = {idx}! {loss} < {min_loss}")
                            min_loss = loss
                            best_model = x
                            if len(os.listdir(x_path))>1:
                                try:
                                    shutil.copytree(x_path, self.config["SaveDir"]+"/best_model/")
                                except FileExistsError:
                                    shutil.rmtree(self.config["SaveDir"]+"/best_model/")
                                    shutil.copytree(x_path, self.config["SaveDir"]+"/best_model/")
                        shutil.rmtree(x_path) #type: ignore
                        nb_receive+=1
                    results = [i['Loss'] for i in population]
                    sorted_indexes = np.argsort(results)[:int(np.floor(n_i / self.eta))]
                    kept_index = [population[i]['Idx'] for i in sorted_indexes]
                    pop = [population[i] for i in sorted_indexes]
                    try:
                        for p in population:
                            if p['Idx'] not in kept_index:
                                try:
                                    os.remove(p['Individual'])
                                except Exception as e:
                                    logger.error(f"Cannot remove {p['Idx']}: {e}")
                    except Exception as e:
                        logger.error(f"This last loop failed with {e}")

                    
            return best_model, min_loss
        
        else:
            logger.info(f"Worker {self.rank} here.")
            stop = True
            while stop:
                action, x, n_epochs, idx = self.comm.recv(source=0, tag=0, status=self.status)
                if action != None:
                    if action == "evaluate":
                        self.trainer.config['NumEpochs'] = n_epochs
                        self.trainer.config['M'] = n_epochs/self.trainer.config['LoopSize']
                        self.trainer.config['Callbacks'].append(self.checkpoint(save_dir="/snapshots", save_top_k=1, max_checkpoints=self.trainer.config['M'], max_epoch=n_epochs))
                        self.trainer.config['Optimizer']= {
                            "Optimizer": "SGD",
                            "Scheduler": {
                                "Scheduler":torch.optim.lr_scheduler.LambdaLR,
                                "Args": {"lr_lambda": lambda it: self.lr(it, self.trainer.config['M'], self.trainer.config['NumEpochs'], self.trainer.config['LearningRate'])}
                            },
                            "LearningRate": 1,
                            }
                        if isinstance(x, str):
                            x_path = x
                            with open(x_path, 'rb') as f:
                                x = pickle.load(f)
                            os.remove(x_path)
                        loss, model = self.trainer.train_and_predict(x, idx=idx)
                        x_path = os.path.join(self.config['SaveDir'], f"{idx}") #type: ignore
                        os.makedirs(x_path, exist_ok=True)
                        if hasattr(model, "save"):
                            model.save(x_path)                        
                        with open(x_path + "/x.pkl", 'wb') as f:
                            pickle.dump(x, f)
                        del x
                        self.comm.send(dest=0, tag=0, obj=[loss, x_path, idx])
                else:
                    logger.info(f'Worker {self.rank} has been stopped') #type: ignore
                    stop = False
