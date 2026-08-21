# DragonSR–PySR Leaderboard Pipeline

This directory contains the experimental pipeline used to compare **DragonSR** and **PySR** on the symbolic regression benchmarks reported in:

> *DragonSR: Directed Acyclic Graph Search forEquation Discovery*

The pipeline provides a common interface for:

* running DragonSR only;
* running PySR only;
* running DragonSR and PySR successively;
* generating synthetic Nguyen and physics benchmark data;
* loading the remote-sensing benchmark data;
* loading a custom CSV dataset;
* adding controlled target noise;
* executing multiple independent runs;
* collecting the run-level results;
* generating an interactive HTML leaderboard.

The complete run-level outputs used in the paper are archived on Zenodo:

**https://doi.org/10.5281/zenodo.21998389**

## Repository structure

```text
leaderboard/
├── main.py                    # Main experiment orchestrator
├── Config.py                  # Default experiment configuration
├── config.txt                 # Example DragonSR configuration
├── config2.txt                # Additional DragonSR configuration
├── config_pysr.txt            # Example PySR configuration
├── requirements.txt           # Python dependencies
├── dataprocessing/            # Dataset generation and loading
├── runner/                    # DragonSR and PySR runners
├── searchspace/               # DragonSR search-space definitions
├── pipeline/                  # Search and post-processing components
└── helpers/                   # Configuration and leaderboard utilities
```

## Installation

Clone the repository and move to the leaderboard directory:

```bash
git clone https://github.com/JulieKeisler/DRAGON.git
cd DRAGON
git checkout dev-elyas
cd leaderboard
```

Install the required Python packages:

```bash
python3 -m pip install -r requirements.txt
```

The requirements install the local DRAGON package together with the dependencies used by the leaderboard pipeline, including PySR.

## Configuration

Experiment settings are defined in `Config.py` and can be overridden at runtime using a text configuration file:

```bash
python3 main.py --config_file_path path/to/config.txt
```

A configuration file may specify, among others:

```text
TARGETS = n4,n5,n6
N_RUNS = 5
INIT_STRATEGIES = random
RANDOM_SEED = 42
NOISE_STD = 0.01

N_ITERATIONS = 10000
K_INIT = 30
MAX_COMPLEXITY = 55
T_PER_LEVEL = 200

SEARCH_LOSS = corr
SUBSAMPLING_RATIO = 1.0

parallel_mode = smart
loss_mode = channel
var_aug = true
add_noise = true
denoiser = gpr
sampling = true
optimize_constants = true
optimizer = dichotomy
```

PySR-specific parameters use the `PySR.` prefix:

```text
PySR.N_ITERATIONS = 10000
PySR.POPULATIONS = 5
PySR.POPULATION_SIZE = 120
PySR.ADD_NOISE = false
PySR.SHOULD_OPTIMIZE_CONSTANTS = true
PySR.MAXSIZE = 30
PySR.BINARY_OPS = +,-,*,/
PySR.UNARY_OPS = log,exp,sin,cos,sqrt,abs
```

The provided configuration files are examples and can be copied and adapted to the desired experimental setting.

## Supported benchmarks

The built-in benchmark loader supports:

* Nguyen functions `n1` to `n12`;
* physics expressions including `hubble`, `newton`, `rydberg`, `idealgas`, `kepler`, `bode`, `schechter`, `leavitt`, and `planck`;
* remote-sensing indices including `ndvi`, `wi2015`, `awei_sh`, `awei_nsh`, `bai`, `bsi`, `evi2`, `mndwi`, `vari`, `savi`, and `nirv`.

The benchmarks evaluated in a given experiment are selected through `TARGETS`.

## Running the experiments

All commands below must be executed from the `leaderboard` directory.

### Run DragonSR and PySR

When no method-selection option is provided, the pipeline runs DragonSR first and PySR second:

```bash
python3 main.py --config_file_path config.txt
```

A fresh combined execution removes an existing `results.json` from the configured output directory before starting.

### Resume a combined experiment

To retain the existing results and execute only missing or failed entries:

```bash
python3 main.py \
  --config_file_path config.txt \
  --continue
```

### Run DragonSR only

```bash
python3 main.py \
  --dragon-only \
  --config_file_path config.txt
```

### Run PySR only

```bash
python3 main.py \
  --pysr-only \
  --config_file_path config_pysr.txt
```

The method-specific modes load an existing `results.json` when available, skip completed entries, and rerun entries identified as failed.

## Running on a custom CSV dataset

A custom CSV file can be supplied with `--data_path`:

```bash
python3 main.py \
  --dragon-only \
  --data_path /path/to/dataset.csv \
  --config_file_path /path/to/config.txt
```

The target specified by `TARGETS` must correspond to a column in the CSV file. All other numeric columns are used as input variables.

For example:

```text
x1,x2,y
0.12,1.42,2.31
0.37,1.08,2.76
0.51,0.83,3.02
```

must be paired with:

```text
TARGETS = y
```

The same external-data mechanism can be used with `--pysr-only` or with the combined DragonSR–PySR execution.

## Noise convention

When target noise is enabled, the noisy target is generated as:

[
y_{\mathrm{noisy}}
==================

y+\mathcal{N}!\left(0,\epsilon,\operatorname{std}(y)\right),
]

where (\epsilon) is set through `NOISE_STD`.

The experiments reported in the paper use:

```text
NOISE_STD = 0
NOISE_STD = 0.01
NOISE_STD = 0.05
NOISE_STD = 0.10
```

The data seed is derived from `RANDOM_SEED` and the run identifier so that independent runs remain reproducible.

## Outputs

By default, run-level results are written to:

```text
leaderboard_runs/results.json
```

The JSON file is updated after each completed run. It contains information such as:

* benchmark and method identifiers;
* run identifier and initialization strategy;
* discovered symbolic expression;
* evaluation loss and metrics;
* runtime;
* search-space information;
* DragonSR DAG and channel information;
* PySR hall-of-fame information.

The pipeline also generates an interactive HTML leaderboard from the collected results.

To rebuild the leaderboard without running new experiments:

```bash
python3 main.py --leaderboard
```

To update missing results for one method before rebuilding it:

```bash
python3 main.py \
  --leaderboard \
  --dragon-only \
  --config_file_path config.txt
```

or:

```bash
python3 main.py \
  --leaderboard \
  --pysr-only \
  --config_file_path config_pysr.txt
```

## Reproducing the reported results

The source code in this directory implements the common experimental pipeline used for both DragonSR and PySR. The exact run-level outputs used to compute the leaderboard reported in the paper are available in the associated Zenodo archive:

**https://doi.org/10.5281/zenodo.21998389**

The Zenodo archive is organized by:

```text
method/
└── noise condition/
    └── benchmark expression/
        └── independent run/
```

This separates the immutable experimental outputs used in the paper from the evolving source code hosted on GitHub.

## Citation

If you use this pipeline or the archived experimental results, please cite the associated paper and Zenodo dataset.
