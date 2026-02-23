import sys
import lib.dragon
sys.modules['dragon'] = sys.modules['lib.dragon']
import os
import sys
import importlib
import graphviz
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import openml

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from dragon.search_space.bricks_variables import mlp_var, identity_var, operations_var, mlp_const_var, dag_var, node_var, activation_var
from dragon.search_space.base_variables import CatVar, Constant, DynamicBlock, IntVar, FloatVar, ArrayVar, Variable
from dragon.search_space.dag_encoding import AdjMatrix, Node
from dragon.search_space.dag_variables import NodeVariable, HpVar
from dragon.search_operators.base_neighborhoods import ArrayInterval, DynamicBlockInterval, CatInterval, ConstantInterval
from dragon.search_operators.dag_neighborhoods import CatHpInterval, NodeInterval
from dragon.utils.plot_functions import draw_cell, load_archi, str_operations

from dragon.utils.plot_functions import expr_to_mini_dag, graph_to_formula

import numpy as np
from dragon.utils.plot_functions import graph_to_formula

df = pd.read_csv(f"data/6000_points.csv")
df = df.drop(columns=['system:index', 'QA60', '.geo'])
df = df.drop(columns=['B8A'])

df['SAVI'] = (df['B8']-df['B4'])/(df['B8']+df['B4']+0.5)*(1.5)

output = 'SAVI' # NDVI, SAVI, BSI, TEST
X = df.drop([output, 'date'], axis=1)
z = df[[output]]

import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Normalize features for XGBoost
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train XGBoost model for feature importance
xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    verbosity=0
)

xgb_model.fit(X_scaled, z.values.ravel())

# Get feature importances
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print("Feature Importance Ranking:")
print(feature_importance)

# Create dictionary with performance scores for each feature
feature_scores_dict = dict(zip(feature_importance['feature'], feature_importance['importance']))
print("\nFeature Scores Dictionary:")
print(feature_scores_dict)



# Select top features (e.g., top 10)
top_n = 10
top_features = feature_importance.head(top_n)['feature'].tolist()
print(f"\nTop {top_n} features:")
print(top_features)

# Use selected features for your symbolic regression
X_selected = X[top_features]
print(f"\nShape of original features: {X.shape}")
print(f"Shape of selected features: {X_selected.shape}")


num_keep_features = 12
top_features = feature_importance.head(num_keep_features)['feature'].tolist()
X.drop(columns=[col for col in X.columns if col not in top_features], inplace=True)

class MetaArchi(nn.Module):
    def __init__(self, args, input_shape):
        super().__init__()
        self.input_shape = input_shape
        assert isinstance(args['Dag'], AdjMatrix), f"The 'Dag' argument should be an 'AdjMatrix'. Got {type(args['Dag'])} instead."
        self.dag = args['Dag']
        self.dag.set(input_shape)


    def forward(self, X):
        return self.dag(X)
    def set_prediction_to_save(self, name, df):
        if hasattr(self, "prediction"):
            self.prediction[name] = df
        else:
            self.prediction = {name: df}
    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        full_path = os.path.join(path, "best_model.pth")
        torch.save(self.state_dict(), full_path)
        if hasattr(self, "prediction"):
            for k in self.prediction.keys():
                self.prediction[k].to_csv(os.path.join(path, f"best_model_{k}_outputs.csv"))


class RegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X.values)
        self.y = torch.FloatTensor(y.values)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_set = RegressionDataset(X, z)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)


from itertools import combinations

# Generate all possible combinations of feature indices
all_feature_combinations = []
num_features = 12 #12
for length in range(1, num_features):
    all_feature_combinations.extend([list(combo) for combo in combinations(range(num_features), length)])


from dragon.search_operators.base_neighborhoods import CatInterval, ConstantInterval
from dragon.search_operators.dag_neighborhoods import EvoDagInterval, HpInterval
from dragon.search_space.base_variables import CatVar, Constant
from dragon.search_space.bricks.basics import Identity
from dragon.search_space.bricks.symbolic_regression import *
from dragon.search_space.dag_variables import EvoDagVariable, HpVar
from dragon.search_space.bricks_variables import (
    operations_var,
    dag_var,
)
from dragon.search_space.base_variables import ArrayVar
from dragon.search_space.dag_encoding import SymbolicNode
from dragon.search_operators.base_neighborhoods import ArrayInterval

# ── Feature importance → CatVar weights ──────────────────────────────
# Build a per-feature probability vector aligned with X.columns,
# then convert it into per-combination weights for the CatVar.
feature_names_list = X.columns.tolist()
num_feat = len(feature_names_list)
feature_probs = [feature_scores_dict.get(col, 1.0 / num_feat)
                 for col in feature_names_list]

combo_weights = SelectFeatures.combination_weights(
    feature_probs, all_feature_combinations
)

# ── Search-space variables ────────────────────────────────────────────
unary_var = HpVar(
    "UnaryOp",
    CatVar(
        "UnaryOpType",
        features=[Identity, Inverse, Negate],
        neighbor=CatInterval()
    ),
    hyperparameters={},
    neighbor=HpInterval()
)
select_features_var = HpVar(
    "SelectFeatures",
    Constant("SelectFeaturesOp", SelectFeatures, neighbor=ConstantInterval()),
    hyperparameters={
        "feature_indices": CatVar(
            "feature_indices",
            features=all_feature_combinations,
            weights=combo_weights,# biased by XGBoost importance
            neighbor=CatInterval()
        )
    },
    neighbor=HpInterval()
)


sum_var = HpVar(
    "Sum",
    Constant("SumOp", SumFeatures, neighbor=ConstantInterval()),
    hyperparameters={},
    neighbor=HpInterval()
)

const_var = HpVar(
    "Constant",
    Constant("ConstantOp", ConstantBrick, neighbor=ConstantInterval()),
    hyperparameters={},
    neighbor=HpInterval()
)


candidate_operations = operations_var(
    "CandidateOperations",
    size=9,
    candidates=[
        select_features_var,
        unary_var,
        sum_var,
        # const_var
    ],
    combiner_features=['add', 'mul'],
    activations=Constant(
        "id",
        value=nn.Identity(),
        neighbor=ConstantInterval()
    ),
    node_type=SymbolicNode
)

dag = EvoDagVariable(
                    label="Dag",
                    operations = candidate_operations,
                    init_complexity=2,
                    neighbor=EvoDagInterval(nb_mutations=1)
                )


search_space = ArrayVar(dag, label="Search Space", neighbor=ArrayInterval())



def correl(pred, true):
    """Calculates the Pearson correlation coefficient"""
    # Convert to numpy if needed
    if isinstance(pred, torch.Tensor):
        pred = pred.numpy()
    if isinstance(true, torch.Tensor):
        true = true.numpy()
    
    # Calculate deviations from mean
    pred_dev = pred - pred.mean()
    true_dev = true - true.mean()
    
    # Calculate covariance
    cov = np.sum(pred_dev * true_dev)
    
    # Calculate standard deviations
    std_pred = np.sqrt(np.sum(pred_dev ** 2))
    std_true = np.sqrt(np.sum(true_dev ** 2))
    
    # Return correlation coefficient
    if std_pred == 0 or std_true == 0:
        return 0.0
    return cov / (std_pred * std_true + 1e-8)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get feature names and number of features dynamically
feature_names = X.columns.tolist()
num_features = X.shape[1]

def eval(args, eps=1e-1, num_features=num_features):
    model = MetaArchi(args, input_shape=(num_features,)).to(device)
    
    # # Optimization
    # rand = np.random.rand()
    # if rand > 0.5:
    #     X_train_full, y_train_full = loader_to_tensors(train_loader, device)
    #     fit_constants_lbfgs(model, X_train_full, y_train_full, steps=50, lr=1.0)

    model.eval()

    all_pred, all_true = [], []

    with torch.no_grad():
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            pred = model(Xb)
            all_pred.append(pred.detach().cpu())
            all_true.append(yb.detach().cpu())
    pred_all = torch.cat(all_pred)
    true_all = torch.cat(all_true)

    mse = np.inf
    selected_c = 0

    if len(pred_all.shape) > 1:
        for c in range(pred_all.shape[-1]):
            loss = 1 -(correl(pred_all[:, c:c+1], true_all))**2
            if loss < mse:
                mse = loss
                selected_c = c
        pred_all = pred_all[:, selected_c]

    pred_all = pred_all.squeeze().numpy()
    true_all = true_all.squeeze().numpy()

    # close_matches = np.sum(np.abs(pred_all - true_all) < eps)
    # fraction_close = close_matches / len(pred_all)
    # var = np.abs(true_all.std() - pred_all.std())
    return model, pred_all, true_all, mse
    

def loss_function(args, idx, eps=1e-1, mse_threshold=0.04, feature_names=feature_names, *kwargs):
    labels = [e.label for e in search_space]
    args = dict(zip(labels, args))

    model, y_pred, y_true, mse = eval(args, eps)

    # Formula extraction disabled to avoid slow sympy.simplify on complex expressions
    # try:
    #     expr = graph_to_formula(
    #         args['Dag'].matrix,
    #         np.asarray(feature_names),
    #         args['Dag'].operations
    #     )
    # except Exception as e:
    #     expr = None
    expr = None

    print(
        f"Idx={idx}, MSE = {mse:.10f}"
        # f"formula={expr}"
    )
    rand = np.random.rand()
    if rand > 0.5: # This is passed since there's no expression calculated
        if expr is not None:
            try:
                mini_dag = expr_to_mini_dag(expr, feature_names)
                args['Dag'] = mini_dag
                model, y_pred, y_true, mse= eval(args, eps)
                expr = graph_to_formula(
                    args['Dag'].matrix,
                    np.asarray(feature_names),
                    args['Dag'].operations
                )
                print(f"Idx={idx}, mini dag MSE={mse:.6f}, formula = {expr}")
            except Exception as e:
                pass
        model.mini_x = [v for v in args.values()]
    df = pd.DataFrame({
        "pred": y_pred,
        "true": y_true
    })
    model.set_prediction_to_save("prediction", df)
    return mse.item(), model





from dragon.search_algorithm.mutant_ucb import Mutant_UCB



# EXPLORATION_THRESHOLD = 0.01   # switch from E=1000 → E=0.01
# FINAL_THRESHOLD = 1e-6         # stop completely — formula found
# T_MAX = 5000                   # max iterations per phase
# K_INIT = 200
# save_dir = "save/NDVI_formula/"

# # Phase 1: Exploration (E=1000) until loss < EXPLORATION_THRESHOLD
# print("=== Phase 1: Exploration (E=1000) ===")
# search_algorithm = Mutant_UCB(
#     search_space=search_space,
#     evaluation=loss_function,
#     T=T_MAX,
#     K=K_INIT,
#     N=1,
#     E=1000,
#     save_dir=save_dir,
#     clean_all=False,
#     verbose=True,
#     loss_threshold=EXPLORATION_THRESHOLD,
# )
# search_algorithm.run()
# print(f"[Exploration done] best loss = {search_algorithm.min_loss:.8f}")

# if search_algorithm.min_loss >= EXPLORATION_THRESHOLD:
#     print(f"Did not reach {EXPLORATION_THRESHOLD} after {T_MAX} iters. Stopping.")
# else:
#     # Phase 2: Exploitation (E=0.01) until loss < FINAL_THRESHOLD
#     pop_size = len(pd.read_csv(os.path.join(save_dir, "computation_file.csv")))
#     T_exploit = pop_size + T_MAX
#     print(f"\n=== Phase 2: Exploitation (E=0.01, {T_MAX} new iters) ===")
#     search_algorithm = Mutant_UCB(
#         search_space=search_space,
#         evaluation=loss_function,
#         T=T_exploit,
#         K=K_INIT,
#         N=1,
#         E=0.01,
#         save_dir=save_dir,
#         clean_all=False,
#         verbose=True,
#         pop_path=save_dir,
#         loss_threshold=FINAL_THRESHOLD,
#     )
#     search_algorithm.run()
#     best = search_algorithm.min_loss
#     if best <= FINAL_THRESHOLD:
#         print(f"\nFormula found! Final loss = {best:.10f}")
#     else:
#         print(f"\nExploitation done. Best loss = {best:.8f} (threshold {FINAL_THRESHOLD} not reached)")



from mpi4py import MPI
import os
import pandas as pd

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

EXPLORATION_THRESHOLD = 0.01
FINAL_THRESHOLD = 1e-6
T_MAX = 5000
K_INIT = 200
save_dir = "save/NDVI_formula/"

# ---------------- Phase 1 ----------------
if rank == 0:
    print("=== Phase 1: Exploration (E=1000) ===", flush=True)

search_algorithm = Mutant_UCB(
    search_space=search_space,
    evaluation=loss_function,
    T=T_MAX,
    K=K_INIT,
    N=1,
    E=1000,
    save_dir=save_dir,
    clean_all=False,
    verbose=(rank == 0),           # avoid log spam
    loss_threshold=EXPLORATION_THRESHOLD,
)

search_algorithm.run()

# Sync + decide next step ONLY on rank 0
comm.Barrier()
best1 = comm.bcast(search_algorithm.min_loss if rank == 0 else None, root=0)

if rank == 0:
    print(f"[Exploration done] best loss = {best1}", flush=True)

go_phase2 = (best1 < EXPLORATION_THRESHOLD)
go_phase2 = comm.bcast(go_phase2, root=0)

# ---------------- Phase 2 ----------------
if not go_phase2:
    if rank == 0:
        print(f"Did not reach {EXPLORATION_THRESHOLD} after {T_MAX} iters. Stopping.", flush=True)
else:
    # Only rank 0 should read files to compute T_exploit, then broadcast it
    if rank == 0:
        pop_size = len(pd.read_csv(os.path.join(save_dir, "computation_file.csv")))
        T_exploit = pop_size + T_MAX
        print(f"\n=== Phase 2: Exploitation (E=0.01, {T_MAX} new iters) ===", flush=True)
    else:
        T_exploit = None

    T_exploit = comm.bcast(T_exploit, root=0)

    search_algorithm = Mutant_UCB(
        search_space=search_space,
        evaluation=loss_function,
        T=T_exploit,
        K=K_INIT,
        N=1,
        E=0.01,
        save_dir=save_dir,
        clean_all=False,
        verbose=(rank == 0),
        pop_path=save_dir,
        loss_threshold=FINAL_THRESHOLD,
    )

    search_algorithm.run()

    comm.Barrier()
    best2 = comm.bcast(search_algorithm.min_loss if rank == 0 else None, root=0)

    if rank == 0:
        if best2 <= FINAL_THRESHOLD:
            print(f"\nFormula found! Final loss = {best2:.10f}", flush=True)
        else:
            print(f"\nExploitation done. Best loss = {best2:.8f} (threshold {FINAL_THRESHOLD} not reached)", flush=True)