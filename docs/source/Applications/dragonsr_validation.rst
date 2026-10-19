DragonSR: validation at the paper's defaults
=============================================

The packaged ``DragonSR`` estimator was run at its default settings, the
configuration of the DragonSR paper, on six laws whose results the paper
reports: NDVI, AWEI\ :sub:`sh`, EVI2, Hubble, Nguyen-4 and Nguyen-11 (x\ :sup:`y`).
There were five runs each, without noise, on the data of the paper's runs with
the same seeds. The reported expressions were judged by reading them, under
the paper's rules, by a different session from the one that wrote the
library.

==========================================  =================  ================
law                                         library, 5 runs    paper, 10 runs
==========================================  =================  ================
NDVI, AWEI\ :sub:`sh`, EVI2, Hubble, N4     5 of 5 each        10 of 10 each
Nguyen-11 (x\ :sup:`y`)                     4 of 5             8 of 10
==========================================  =================  ================

On the 30 shared seeds the two agree on 29; in the one disagreement the
library reports a ratio of polynomials that follows x\ :sup:`y` to 0.05% but is not
the law. Each run spends the same effort as the paper's runs (19,937-20,014
evaluations). With five runs per law, the result is *consistent with* the
paper's runs; it does not reproduce the paper's tables. Data, verdicts and
scripts are in ``validation/dragonsr_paper_defaults/`` of the repository.
