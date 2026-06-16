"""
Definition of all config parameters
"""

import os

# ══════════════════════════════════════════════════════════════════════════════
#  PATHS
# ══════════════════════════════════════════════════════════════════════════════

class Paths:
    REMOTE_DATA_CSV   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "6000_points.csv"))
    EXTERNAL_DATA_CSV = ""  # if non-empty, load target directly from this CSV instead of the built-in remote-sensing loader
    OUTPUT_DIR        = "leaderboard_runs"
    HTML_OUTPUT       = "dragonfsr_leaderboard_v2.html"
    LOG_SUFFIX        = "_found_formulas.txt"


# ══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT
# ══════════════════════════════════════════════════════════════════════════════

class Experiment:
    TARGETS = [
        "n4", "n5", "n6", "n7", "n8", "n9", "n10", "n11", "n12",
        "newton", "rydberg", "idealgas", "kepler", "schechter", "bode", "leavitt", "planck", #"hubble",
        "wi2015", "awei_sh", "bai", "ndvi", "savi", "bsi", "evi2", "mndwi", "vari", "nirv",
    ]
    N_RUNS           = 2 # todo: increase back to 5 when all strategies are defined
    INIT_STRATEGIES  = ["random", "diverse", "xgboost", "warmstart", "adversarial"] #todo: define other startegies ("xgboost", "warmstart", "adversarial")
    RANDOM_SEED      = 42
    N_TOP_FEATURES   = 10
    N_SYNTH_SAMPLES  = 6000
    NOISE_STD        = 0.01


# ══════════════════════════════════════════════════════════════════════════════
#  ALIGNMENT LOSS
# ══════════════════════════════════════════════════════════════════════════════

class Loss:
    KIND             = "mse"   # "mse" | "huber"
    HUBER_DELTA_FRAC = 0.20


# ══════════════════════════════════════════════════════════════════════════════
#  OLS POST-PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

class OLS:
    PARSIMONY_REL_TOL = 0.02
    COMPLEXITY_MAX    = None
    RAT_MAX_DEGREE    = 2
    RAT_MAX_BASIS_CH  = 3
    RAT_MAX_FEATURES  = 4


# ══════════════════════════════════════════════════════════════════════════════
#  DRAGON
# ══════════════════════════════════════════════════════════════════════════════

class Dragon:
    N_ITERATIONS   = 10_000
    K_INIT         = 500
    MAX_COMPLEXITY = 10
    MAX_NODES      = 15
    T_PER_LEVEL    = 1_000
    LOSS_THRESHOLD = 1e-30

    SPAR_OP_GROUPS = {
        "all":        ["select", "unary", "power", "ln", "exp", "sin", "cos"],
        "alg":        ["select", "unary", "power"],
        "alg_trig":   ["select", "unary", "power", "sin", "cos"],
        "alg_explog": ["select", "unary", "power", "ln", "exp"],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  PYSR
# ══════════════════════════════════════════════════════════════════════════════

class PySR:
    N_ITERATIONS    = 1_000
    POPULATIONS     = 5
    POPULATION_SIZE = 120
    MAXSIZE         = 30
    JULIA_PROJECT   = "/Users/elyaschikhaoui/Desktop/dragon/.dragonenv/julia_env"
    BINARY_OPS      = ["+", "-", "*", "/"]
    UNARY_OPS       = ["log", "exp", "sin", "cos", "sqrt", "abs"]


# ══════════════════════════════════════════════════════════════════════════════
#  MC-DROPOUT
# ══════════════════════════════════════════════════════════════════════════════

class MCDropout:
    ENABLED   = False
    N_FORWARD = 50
    DROPOUT_P = 0.15
    N_EPOCHS  = 300
    HIDDEN    = 64


# ══════════════════════════════════════════════════════════════════════════════
#  SAMPLING
# ══════════════════════════════════════════════════════════════════════════════

class Sampling:
    ENABLED = True
    SUBSAMPLING_RATIO = 0.1


# ══════════════════════════════════════════════════════════════════════════════
#  METHOD REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

_ALL_OPS = ["select", "unary", "power", "ln", "exp", "sin", "cos"]

DRAGON_METHODS = [
    {
        "id":                 "spar_denoise",
        "description":        "DragonSR — smart-parallel + stochastic subsampling denoising (SSD).",
        "operators":          _ALL_OPS,
        "parallel_n":         1,
        "parallel_mode":      "smart", # "none" | "smart"
        "loss_mode":          "full", # "ols" | "channel"
        "var_aug":            True,
        "add_noise":          True,
        "denoiser":           "lingam", # "gpr" or "lingam" or None
        "sampling":           Sampling.ENABLED,
        "mc_dropout":         MCDropout.ENABLED,
    },
    # {
    #     "id":          "allops",
    #     "description": "DragonSR — reference method (all ops, full OLS, var-aug, no noise)",
    #     "operators":   ["select", "unary", "power", "ln", "exp", "sin", "cos"], 
    #     "parallel_N":  1,
    #     "loss_mode":   "full",
    #     "var_aug":     True,
    #     "add_noise":   False,
    # },
    # {
    #     "id":          "spar",
    #     "description": ("DragonSR — smart-parallel: 4 op-subset streams "
    #                     "(all / alg / alg+trig / alg+exp,ln) run in parallel "
    #                     "via ThreadPoolExecutor; best loss wins."),
    #     "operators":   ["select", "unary", "power", "ln", "exp", "sin", "cos"],
    #     "parallel_N":  1,
    #     "loss_mode":   "full",
    #     "var_aug":     True,
    #     "add_noise":   False,
    #     "parallel_mode": "smart",
    # },
    # {
    #     "id":          "boosted_spar",
    #     "description": ("DragonSR — boosted smart-parallel: same 4 op-subset "
    #                     "streams as 'spar' run concurrently, then their "
    #                     "winning predictions ŷ_stream are stacked and "
    #                     "meta-combined via sparse-OLS / nested-OLS / "
    #                     "poly-rational-OLS (whichever fits best)."),
    #     "operators":   ["select", "unary", "power", "ln", "exp", "sin", "cos"],
    #     "parallel_N":  1,
    #     "loss_mode":   "full",
    #     "var_aug":     True,
    #     "add_noise":   False,
    #     "parallel_mode": "smart",
    #     "boosted":        True,
    # },
    #  {
    #     "id":          "noolsratn",
    #     "description": "Ablation of allops — NO OLS / rat / nested (channel-only loss)",
    #     "operators":   ["select", "unary", "power", "ln", "exp", "sin", "cos"],
    #     "parallel_N":  1,
    #     "loss_mode":   "channel",
    #     "var_aug":     True,
    #     "add_noise":   False,
    # },
    # {
    #     "id":          "allops_const",
    #     "description": "DragonSR — +ConstantBrick (Adam-optimized constants)",
    #     "operators":   ["select", "unary", "power", "ln", "exp", "sin", "cos", "const"],
    #     "parallel_N":  1,
    #     "loss_mode":   "channel",
    #     "var_aug":     True,
    #     "add_noise":   False,
    #     "optimize_constants": True,
    # }
]