from zellij.core.search_space import MixedSearchspace
from zellij.strategies import Genetic_algorithm
from zellij.strategies.simulated_annealing import Simulated_annealing
from zellij.strategies.tools import MulExponential

from zellij.utils.neighborhoods import Intervals
from zellij.utils.operators import DeapTournament

from lib.evodags.search_algorithm.neighborhoods_operators import HierarchicalNNMutation, NNMutation
from lib.evodags.search_algorithm.other_operators import SelBestWoDuplicate, Random
from lib.evodags.search_algorithm.variation_operators import DAGTwoPoint


def problem_configuration(config, net, loss):
    if config["Neighborhood"] == "Hierarchical":
        assert "FChange" in config, "Hierarchical mutation requires Frequency change (FChange) attribute"
        mutation = HierarchicalNNMutation(config["MutationRate"], change=config["FChange"])
    else:
        mutation = NNMutation(config['MutationRate'])
    search_space = MixedSearchspace(net,
                                  loss=loss,
                                  neighbor=Intervals(),
                                  mutation=mutation,
                                  selection=DeapTournament(max(int(config["PopSize"] / config["TournamentRate"]), 1)),
                                  crossover=DAGTwoPoint(),
                                  random=Random(0.1),
                                  bestSel=SelBestWoDuplicate()
                                  )
    if config["MetaHeuristic"] == "GA":
        search_algorithm = Genetic_algorithm(search_space, 1000000, pop_size=config["PopSize"],
                                             generation=config["Generations"],
                                             elitism=config["ElitismRate"])

    elif config["MetaHeuristic"] == "SA":
        cooling = MulExponential(0.85, 100, 2, 3)
        search_algorithm = Simulated_annealing(search_space, 1000000, cooling, max_iter=20)
    elif not isinstance(config["MetaHeuristic"], str):
        search_algorithm = config['MetaHeuristic'](search_space, 1000000, pop_size=config["PopSize"],
                                             generation=config["Generations"],
                                             elitism=config["ElitismRate"], random=config["RandomRate"],
                                             best_selection=SelBestWoDuplicate)
    return search_space, search_algorithm
