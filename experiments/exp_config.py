from experiments.monash_archive.datasets_configs import dataset_configs

exp_config = {
    "GlobalSeed": 100,
    "Generations": 2,
    "PopSize": 4,
    "DatasetConfig": dataset_configs,
    "SPSize": 5,
    "FChange": ["large" for _ in range(1)] + ["local" for _ in range(2)],
    "MutationRate": 0.7,
    "TournamentRate": 10,
    "ElitismRate": 0.1,
    "RandomRate": 0.1,

}