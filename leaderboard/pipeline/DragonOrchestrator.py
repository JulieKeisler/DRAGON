from __future__ import annotations

import os
import random
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from itertools import combinations
from pathlib import Path

from Config import Dragon as _CfgDragon
from dataprocessing.Denoise import MCDropoutWeighter, SamplingContext
from dataprocessing.Features import CombinationBuilder
from helpers.Helper import _extract_best_formula_from_log


# local helper path
sys.path.insert(0, str(Path(os.path.abspath(os.path.dirname(os.path.abspath(__file__))))))

# leaderboard path
leaderboard_root = Path(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
sys.path.insert(0, str(leaderboard_root))

# DRAGON root path (for lib.dragon)
dragon_root = Path(os.path.abspath(os.path.join(leaderboard_root, "..")))
sys.path.insert(0, str(dragon_root))
import lib.dragon
sys.modules["dragon"] = sys.modules["lib.dragon"]

from dragon.search_algorithm.mutant_ucb import Mutant_UCB
from dragon.search_space.dag_encoding import AdjMatrix, SymbolicNode
from dragon.search_space.bricks.basics import Identity
from dragon.search_space.bricks.symbolic_regression import (
	SelectFeatures, Inverse, Negate, Power, SumFeatures, ConstantBrick,
)


class DragonOrchestrator:
	"""Pipeline orchestrator for DRAGON setup, seeding, and search execution."""

	@staticmethod
	def setup_searcher(method_cfg, X_sel, y, feature_names, feat_scores, log_path, seed, device):
		from torch.utils.data import DataLoader
		from runner.Dragon import DragonSearcher, RegressionDataset
		from searchspace.SearchSpace import SearchSpaceBuilder

		method_id = method_cfg["id"]

		loader = DataLoader(RegressionDataset(X_sel, y), batch_size=32, shuffle=True)
		all_combos = CombinationBuilder().build(feature_names, feat_scores)
		search_space, dag = SearchSpaceBuilder(
			feature_names, feat_scores, method_cfg["operators"]
		).build(all_combos)
		sampling = SamplingContext.get_sampling(method_cfg, X_sel, y)
		sample_weights = MCDropoutWeighter.get_sample_weights(method_cfg, X_sel, y, seed, method_id)

		searcher = DragonSearcher(
			search_space,
			loader,
			device,
			X_sel.shape[1],
			feature_names,
			log_path,
			loss_mode=method_cfg.get("loss_mode", "full"),
			optimize_constants=method_cfg.get("optimize_constants", False),
			subsample_ratio=sampling.subsample_ratio,
			X_np=sampling.X_np,
			y_np=sampling.y_np,
			sample_weights=sample_weights,
		)

		return searcher, dag, search_space

	@staticmethod
	def build_random_seed_dags(feature_names, operator_keys, strategy=None, seed=0, n_seeds=200):
		"""Build a population of seed DAGs for the 'diverse' init strategy."""
		if strategy != "diverse":
			return None
		rng = random.Random(seed)
		if not feature_names:
			return None

		op_pool = [
			("Identity", Identity, {}, nn.Identity()),
			("Inverse", Inverse, {}, nn.Identity()),
			("Negate", Negate, {}, nn.Identity()),
			("SumFeatures", SumFeatures, {}, nn.Identity()),
			*[(f"Power_{exp}", Power, {"exponent": exp}, nn.Identity())
			  for exp in (-3, -2, -1, 1, 2, 3)],
			*[(f"Sel_{i}", SelectFeatures, {"indices": [i]}, nn.Identity())
			  for i in range(len(feature_names))],
			*[(f"Sel_{i}_{j}", SelectFeatures, {"indices": [i, j]}, nn.Identity())
			  for i, j in combinations(range(len(feature_names)), 2)],
		]
		if "const" in operator_keys:
			op_pool.append(("Const", ConstantBrick, {}, nn.Identity()))

		combiner_patterns = (
			lambda k: "add",
			lambda k: "mul",
			lambda k: "add" if k % 2 == 0 else "mul",
			lambda k: "mul" if k % 2 == 0 else "add",
		)
		topologies = ("chain", "fan", "skip", "rand")

		seed_dags = []
		for s in range(n_seeds):
			size = rng.randint(3, 7)
			topo = topologies[s % len(topologies)]
			comb_fn = combiner_patterns[s % len(combiner_patterns)]
			nodes = [
				SymbolicNode(
					combiner=comb_fn(k),
					operation=op_cls,
					hp=dict(hp),
					activation=act,
				)
				for k in range(size)
				for _, op_cls, hp, act in [op_pool[rng.randrange(len(op_pool))]]
			]

			M = np.zeros((size, size), dtype=int)
			if topo in ("chain", "skip", "rand"):
				M[np.arange(size - 1), np.arange(1, size)] = 1
			if topo == "fan":
				M[0, 1:] = 1
			if topo == "skip":
				M[np.arange(size - 2), np.arange(2, size)] = 1
			if topo == "rand":
				for i in range(size):
					for j in range(i + 1, size):
						if rng.random() < 0.3:
							M[i, j] = 1
			try:
				seed_dags.append(AdjMatrix(operations=nodes, matrix=M))
			except Exception:
				continue

		return [[m] for m in seed_dags] if seed_dags else None

	@staticmethod
	def run_search(method_cfg, search_space, dag, searcher, save_dir, seed_models, _max_iters):
		parallel_N = method_cfg.get("parallel_N", 1)
		os.makedirs(save_dir, exist_ok=True)

		def _make_sa(T, clean, extra=None):
			kw = dict(
				search_space=search_space,
				evaluation=searcher,
				T=T,
				K=_CfgDragon.K_INIT,
				N=parallel_N,
				E=1000,
				save_dir=save_dir,
				clean_all=clean,
				verbose=True,
				loss_threshold=_CfgDragon.LOSS_THRESHOLD,
				**(extra or {}),
			)
			if clean and seed_models is not None:
				kw["models"] = seed_models
			return Mutant_UCB(**kw)

		_iter_cap = _max_iters if _max_iters is not None else _CfgDragon.N_ITERATIONS

		if not method_cfg.get("curriculum", False):
			sa = _make_sa(_iter_cap, clean=True)
			sa.run()
			return sa.min_loss

		global_best = np.inf
		total_iters = 0
		budget_mode = method_cfg.get("budget_mode", "default")
		for complexity in range(1, _CfgDragon.MAX_COMPLEXITY + 1):
			remaining = _iter_cap - total_iters
			if remaining <= 0:
				break
			dag.complexity = complexity
			clean = complexity == 1
			csv_path = os.path.join(save_dir, "computation_file.csv")
			pop = len(pd.read_csv(csv_path)) if not clean and os.path.exists(csv_path) else 0
			extra = {} if clean or budget_mode == "complexity" else {"pop_path": save_dir}
			T = min(pop + complexity * _CfgDragon.T_PER_LEVEL, remaining)
			sa = _make_sa(T, clean, extra)
			sa.run()
			try:
				total_iters = len(pd.read_csv(csv_path))
			except Exception:
				total_iters += T
			global_best = min(global_best, sa.min_loss)
			if global_best <= _CfgDragon.LOSS_THRESHOLD:
				break
		return global_best

	@classmethod
	def run_worker_pipeline(
		cls,
		method_cfg,
		X_sel,
		y,
		feature_names,
		feat_scores,
		log_path,
		seed,
		device,
		run_id,
		strategy,
		save_dir,
		_max_iters=None,
	):
		searcher, dag, search_space = cls.setup_searcher(
			method_cfg, X_sel, y, feature_names, feat_scores, log_path, seed, device
		)
		loss_state = searcher.state

		seed_models = cls.build_random_seed_dags(
			feature_names, method_cfg["operators"], strategy=strategy, seed=run_id
		)

		best_loss = cls.run_search(
			method_cfg, search_space, dag, searcher, save_dir, seed_models, _max_iters
		)
		searcher.finalize_ols_postprocessing()

		best_formula = loss_state.get("best_formula", "N/A")
		if best_formula == "N/A":
			best_formula = _extract_best_formula_from_log(log_path)

		return best_loss, best_formula, loss_state
