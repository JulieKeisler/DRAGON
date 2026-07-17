"""Symbolic regression utilities for DRAGON.

This package contains all DRAGON-specific symbolic regression components:

Submodules
----------
- ``dag_to_formula`` — DAG extraction, SymPy compilation, formula formatting
- ``formula_extraction`` — OLS analysis dict → formula strings
- ``dag_inspector`` — DAG structure introspection and SVG/text export
- ``denoise`` — GPR/LinGAM denoisers, MC-Dropout, noise injection
- ``features_selection`` — Feature selectors, VarAugmentor, CombinationBuilder
- ``simplification`` — Composition penalty for nested same-family functions
- ``loss_function/`` — Search losses, OLS pipeline, fitting, selection
"""
