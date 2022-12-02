import warnings

from framework.operators.neighborhoods_operators import HierarchicalNNMutation
from framework.operators.other_operators import selBestWODuplicates
from framework.operators.variation_operators import DAGTwoPoint

warnings.filterwarnings("ignore")

import torch
import logging
from zellij.core.loss_func import Loss
from zellij.core.search_space import HpoSearchspace
from zellij.strategies.genetic_algorithm import Genetic_algorithm
from zellij.utils.neighborhoods import Intervals
from zellij.utils.operators import DeapTournament
from utils.tools import argparser, set_seed
from experiments.exp_config import exp_config
from experiments.monash_archive.dataset import gluonts_dataset
from experiments.monash_archive.search_space import monash_search_space
from experiments.monash_archive.training import GluontsNet

if __name__ == "__main__":
    try:
        log_gluonts = logging.getLogger("gluonts")
        log_gluonts.setLevel('CRITICAL')
        log_mxnet = logging.getLogger("pytorch_lightning")
        log_mxnet.setLevel('CRITICAL')
        # Loss Construction
        CLI = argparser()
        args = CLI.parse_args()
        dataset = args.benchmark[0]
        config = exp_config["DatasetConfig"][dataset]
        set_seed(exp_config['GlobalSeed'])
        n_gpu = torch.cuda.device_count()
        if n_gpu > 0:
            config["Device"] = "cuda:0"
        config['SaveDir'] = config['PathName'] + "_"
        config['SPSize'] = exp_config['SPSize']
        train_ds, test_ds, config = gluonts_dataset(config)
        model = GluontsNet(train_ds, test_ds, config)
        net = monash_search_space(config)
        labels = [e.label for e in net]

        loss = Loss(verbose=False, save=True, separator=';', labels=labels)(model.get_nn_forecast)
        pop = exp_config['PopSize']
        change = exp_config['FChange']
        search_space = HpoSearchspace(net,
                                      loss=loss,
                                      neighbor=Intervals(),
                                      mutation=HierarchicalNNMutation(exp_config["MutationRate"], change=change),
                                      selection=DeapTournament(max(int(pop / exp_config["TournamentRate"]), 1)),
                                      crossover=DAGTwoPoint(),
                                      )
        ga = Genetic_algorithm(search_space, 1000000, pop_size=pop, generation=exp_config["Generations"],
                               elitism=exp_config["ElitismRate"], random=exp_config["RandomRate"],
                               best_selection=selBestWODuplicates)
        ga.run()
        ga.show(save=True)

        loss.stop()
    except Exception as e:
        raise e
