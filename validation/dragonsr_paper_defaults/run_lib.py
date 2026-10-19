"""Library validation: the packaged DragonSR estimator with the paper's defaults.

    python run_lib.py <target> <run_id> <out_dir>

Data: the frozen experiment loader (leaderboard/dataprocessing/Data.py,
DatasetLoader().load(target, run_id), seed 42 + run_id, 6,000 synthetic
observations), so each run sees exactly the data of the paper's run with the
same run_id at eps = 0.  Estimator: dragon.search_algorithm.dragonsr.DragonSR
from branch dragonsr (commit given in the output), every argument at its
default (the paper's configuration) except seed = 42 + run_id, as the paper.
Writes <out_dir>/<target>_run<run_id>.json.
"""
import json, os, sys, time, logging, warnings
LIB = "/scratch/jkeisler/dragonsr_lib/lib"
HARNESS = "/scratch/jkeisler/EvoDagsAutoDL/leaderboard"
sys.path.insert(0, LIB)
sys.path.append(HARNESS)
logging.disable(logging.WARNING); warnings.filterwarnings("ignore")
import dragon
assert os.path.dirname(dragon.__file__).startswith(LIB), dragon.__file__
from dragon.search_algorithm.dragonsr import DragonSR
from dataprocessing.Data import DatasetLoader

target, run_id, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
X, y = DatasetLoader().load(target, run_id=run_id)
t0 = time.time()
m = DragonSR(seed=42 + run_id).fit(X, y)
rec = {"target": target, "run": run_id, "formula": m.formula_, "winner_type": m.winner_type_,
       "stream": m.stream_, "loss": m.loss_, "n_evaluations": m.n_evaluations_,
       "time_s": time.time() - t0, "feature_names": m.feature_names_,
       "streams": [{k: v for k, v in s.items() if k != "formula"} | {"formula": s["formula"]}
                   for s in m.streams_],
       "library_commit": open(os.path.join(LIB, "..", "COMMIT")).read().strip()}
os.makedirs(out, exist_ok=True)
json.dump(rec, open(os.path.join(out, f"{target}_run{run_id}.json"), "w"), indent=1, default=float)
print(target, run_id, m.winner_type_, m.n_evaluations_, "%.0fs" % rec["time_s"], m.formula_[:200])
