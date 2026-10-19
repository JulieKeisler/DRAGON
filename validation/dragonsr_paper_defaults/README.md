# DragonSR library validation: paper defaults

Does the packaged estimator, `dragon.search_algorithm.dragonsr.DragonSR`, at
its default settings (the configuration of the DragonSR paper), behave like the
code that produced the paper?

**Design.** Six laws with known results in the paper (NDVI, AWEI_sh, EVI2,
Hubble, Nguyen-4 and Nguyen-11 = x^y), five runs each, without noise, on exactly
the data of the paper's runs with the same seeds (the frozen loader,
`DatasetLoader().load(target, run_id)`, seed 42 + run_id), library commit
`681afa224861f1064cbee0ce9ef41f0c7987aa21`. Run on the CLEPS cluster
(`libval_array.sh`, `run_lib.py`; job 5606081, 18-19 October 2026).

**Judging.** Each reported expression was judged by reading it, under the
paper's rule set (algebraic equivalence; the residual check only refutes), by
the experiment session of the project, not by the session that wrote the
library. Verdicts: `verdicts_libval.csv` (md5 `e67edad275dd46273cff892a36841e0f`).

**Result.**

| law | library, 5 runs | paper, 10 runs |
|---|---|---|
| NDVI, AWEI_sh, EVI2, Hubble, Nguyen-4 | 5/5 each | 10/10 each |
| Nguyen-11 (x^y) | 4/5 | 8/10 |

On the 30 shared seeds the library and the paper's runs agree on 29. The one
disagreement is Nguyen-11 run 0. There the library reports a ratio of
polynomials in y(x-1) that follows x^y to 0.05% of RMS(y) over the sampled box.
It contains no logarithm and no exponential, so it is not the law; the paper's
run with that seed reports the law. Every run spent 19,937-20,014 evaluations,
against 19,922-20,040 in the paper's runs. None of the 29 recoveries used the 1%
dropped-term allowance.

**What this does and does not show.** Five runs per law cannot separate rates
closer than about 20 percentage points, and the six laws were chosen, not
sampled. They were chosen to span the kinds of law in the paper (two ratio
indices, a multi-band linear index, a one-variable proportionality, a polynomial,
and x^y, the one transcendental law the paper's own runs miss), not as the laws
the method does best on. The check establishes that the packaged estimator, at its own
defaults, finds the same laws on the same problems with the same effort, at
rates consistent with the paper's runs. It does not reproduce the paper's
tables. Two observations: the same law is often printed in very different
forms (NDVI's five recoveries are five different-looking expressions), and a
run is not reproducible bit for bit even with the same seed, because the
twelve parallel searches share one random generator.

**Files.** `results/` holds the 30 run outputs (`<target>_run<r>.json`).
`roll_libval.py` collects them into rows, `libval_paired.py` pairs them with the
paper's verdicts, and both have cluster paths at the top to repoint.
