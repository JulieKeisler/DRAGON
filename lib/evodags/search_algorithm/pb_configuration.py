from zellij.core.search_space import MixedSearchspace
from zellij.strategies import Genetic_algorithm
from zellij.strategies.simulated_annealing import Simulated_annealing
from zellij.strategies.bayesian_optimization import Bayesian_optimization
from zellij.utils.neighborhoods import Intervals
from zellij.utils.operators import DeapTournament
from zellij.utils import Continuous

from evodags.search_algorithm.neighborhoods_operators import HierarchicalNNMutation, NNMutation
from evodags.search_algorithm.other_operators import SelBestWoDuplicate, Random
from evodags.search_algorithm.variation_operators import DAGTwoPoint
from evodags.utils.tools import logger

def problem_configuration(config, net, loss):
    args = {}
    if "Neighborhood" in config:
        assert 'MutationRate' in config, "Remove Neighborhood from config if you do not plan to use a mutation operator."
        if config["Neighborhood"] == "Hierarchical":
            assert "FChange" in config, "Hierarchical mutation requires Frequency change (FChange) attribute"
            mutation = HierarchicalNNMutation(config["MutationRate"], change=config["FChange"])
        else:
            mutation = NNMutation(config['MutationRate'])
        args["mutation"]= mutation
    if "TournamentRate" in config:
        assert "PopSize" in config, "DeapTournament requires a population size."
        args["selection"] = DeapTournament(max(int(config["PopSize"] / config["TournamentRate"]), 1))
    if "RandomRate" in config:
        args["random"] = Random(config['RandomRate'])

    search_space = MixedSearchspace(net,
                                loss=loss,
                                neighbor=Intervals(),
                                crossover=DAGTwoPoint(),
                                bestSel=SelBestWoDuplicate(),
                                to_continuous=Continuous(),
                                **args
                                )
    args = {}
    if "InitPopulation" in config:
        args["filename"]=config["InitPopulation"]
    if "Generations" in config:
        args["generation"] = config["Generations"]
    if "PopSize" in config:
        args["pop_size"] = config["PopSize"]
    if "ElitismRate" in config:
        args["elitism"] = config["ElitismRate"]
    if "Cooling" in config:
        args["cooling"] = config["Cooling"]
    if "MaxIter" in config:
        args["max_iter"] = config["MaxIter"]
    if "Acquisition" in config:
        args["acquisition"] = config["Acquisition"]
    if "MHDevice" in config:
        if config["Device"] == "cpu":
            args["gpu"] = False
        else:
            args["gpu"] = True
    if "QBayesian" in config:
        args["q"] = config["QBayesian"]
    if "ExpSize" in config:
        args["exp_size"] = config["ExpSize"]
    if "GammaBandits" in config:
        args["gamma"] = config["GammaBandits"]
    if "Kernel" in config:
        args["kernel"] = config["Kernel"]
    if "EBandits" in config:
        args["E"] = config["EBandits"]
    
    if config["MetaHeuristic"] == "GA":
        config["MetaHeuristic"] =  Genetic_algorithm
    elif config["MetaHeuristic"] == "SA":
        config["MetaHeuristic"] =  Simulated_annealing
    elif config["MetaHeuristic"] == "BO":
        config["MetaHeuristic"] = Bayesian_optimization
    search_algorithm = config['MetaHeuristic'](search_space, 1000000, **args)
    return search_space, search_algorithm