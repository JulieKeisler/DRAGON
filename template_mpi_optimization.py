import argparse
import warnings
from lib.evodags.search_algorithm.pb_configuration import problem_configuration

warnings.filterwarnings("ignore")

import torch
from mpi4py import MPI
import logging
from zellij.core.loss_func import Loss
from utils.tools import set_seed, logger
from experiments.exp_config import exp_config
from experiments.monash_archive.dataset import gluonts_dataset
from experiments.monash_archive.search_space import monash_search_space
from experiments.monash_archive.training import GluontsNet

if __name__ == "__main__":
    try:
        ############ GENERAL SETUP ############

        # Remove unnecessary logs from imported packages
        log_gluonts = logging.getLogger("gluonts")
        log_gluonts.setLevel('CRITICAL')
        log_mxnet = logging.getLogger("pytorch_lightning")
        log_mxnet.setLevel('CRITICAL')

        # Args parser for experiment
        parser = argparse.ArgumentParser(description='Template Optimization')
        parser.add_argument('--dataset', type=str, required=False, default='m4_daily', help='dataset name')
        parser.add_argument('--seed', type=int, required=False, default=0, help='general seed for the experiment')
        args = parser.parse_args()

        # Multiprocessing setup
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        p = comm.Get_size()

        # Load and set config for the chosen dataset
        dataset = args.dataset
        config = exp_config["DatasetConfig"][dataset]
        n_gpu = torch.cuda.device_count()
        if n_gpu > 0:
            config["Device"] = "cuda:0"
        config['SaveDir'] = config['PathName'] + "_"
        config['SPSize'] = exp_config['SPSize']
        # Set seed for general reproducibility
        set_seed(args.seed)

        logger.info(f'This is MPI optimization script on the dataset {args.dataset}, with {args.seed} as the experiment seed.')
        #######################################

        # Load Dataset
        train_ds, test_ds, config = gluonts_dataset(config)

        # Create NN training functions
        model = GluontsNet(train_ds, test_ds, config)
        # Search space arguments
        net = monash_search_space(config)
        labels = [e.label for e in net]
        # Objective Loss
        loss = Loss(MPI=True, verbose=False, save=True)(model.get_nn_forecast)
        if loss.master:
            # Generate optimization problem
            search_space, search_algorithm = problem_configuration(exp_config, net, loss)
            if exp_config["MetaHeuristic"] == "GA":
                search_algorithm.run(n_process=p - 1)
            elif exp_config["MetaHeuristic"] == "SA":
                x0 = search_space.random_point()
                y0 = search_space.loss(x0)
                search_algorithm.run(X0=x0, Y0=y0, n_process=p - 1)
            elif not isinstance(exp_config["MetaHeuristic"], str):
                search_algorithm.run(n_process=p - 1)
            search_algorithm.show(save=True)
            loss.stop()
    except Exception as e:
        raise e
