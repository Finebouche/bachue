# bachue

*Bioplausible Adaptation of Compact Higher-order Uniform Expansions.*

Learning rules and readouts of order 2 for reservoirs, studied on the prediction of the next phrase of canary songs.
The code was copied from [canopy](https://github.com/Finebouche/canopy) on 2026-11-19 (package `canopy` renamed
`bachue`); canopy keeps the other canary tasks (syllable classification, syllable generation, phrase certainty).

## Contents

The package follows the layout of [hag](https://github.com/Finebouche/hag): the hyperparameters, the final
evaluations and the analyses are separate.

- `bachue/datasets/`: the canary songs (`load_categorical_forecasting.py`; annotations tracked, audio and MFCC caches not tracked, in
  `bachue/datasets/marron1/`) and the benchmarks of HAG's evaluations (`benchmarks.py`: `load("JapaneseVowels")`, ...),
  with the loaders and the preparation copied from `hag.datasets` (`load_*.py`, `pipeline.py`, `preprocessing.py`,
  `spectral_decomposition.py`, with the MFCC of `mfcc_configs.py` merged into it; without `peak_centered_decomposition.py` and its representation
  "custom"). The datasets are downloaded into `bachue/datasets/<name>/` at their first use (not tracked).
- `bachue/models/`: the reservoirs, from hag (`reservoirs.py`); the readouts trained online, `readout_rules/` (LMS and
  NLMS, recursive least squares with the bias as the weight of a constant input, the exploratory Hebbian rule and its
  versions, their dual forms on the monomials, and `training.py`: a readout trained by a rule from its label
  `<rule>`, `<rule>:2` or `<rule>:<selection>`); the generalized readout of order 2 (`generalized_readout.py`); the
  selections of products of degree 2 (`product_selection.py`: random pairs, principal components, linear scores,
  published alternatives).
- `bachue/tasks/`: the prediction of the next phrase (`next_phrase.py`: prediction points, readouts, metrics), the
  phrase boundaries, and the classification of instances.
- `bachue/hpo/`: the hyperparameters. One file per study of the reservoirs (Optuna): `readout_window_study.py` (the readout:
  last, mean, tail, tail_mean, all; `HAG_CANARY_HPO_READOUT`), `learning_rules_study.py` (LMS, NLMS, RLS, the versions of EH,
  and all the monomials of order 2: `ridge:2`, `nlms:2`, `neh:2`; `HAG_CANARY_HPO_RULE`), `partial_order2_study.py`
  (a rule on selected products: `nlms:pca:64`, `ridge:random_pairs:2080`, ...), their shared parts in `studies.py`
  (search space of the reservoirs, run of a study). The other modules choose the hyperparameters of the readouts by
  cross-validation on the train set (`ridge.py`, `readout_rules.py`, `monomial_selection.py`, ...).
- `bachue/performances/`: the final evaluations on the test set, one module per campaign (`next_phrase.py`,
  `readout_rules.py`, `monomial_selection.py`, `monomial_selection_rules.py`, `benchmark_rules.py` for the datasets of
  hag: `python -m bachue.performances.hag_benchmarks_evaluation JapaneseVowels random_ee 10`).
- `bachue/analysis/`: the paired comparisons of the results and the diagnostics of the selections.
- `bachue/settings.py`: the task, its variants and the names of the studies, from the variables of the environment.
- `slurm/`: job scripts for PlaFRIM (`submit_hpo_rules.sh`: the studies of the rules, `DATASETS="marron1 JapaneseVowels"`;
  `merge_studies.sh`: the parts of the jobs merged into the databases). `tests/`: unit tests.
- `outputs/`: one folder per study, `readout_window/`, `learning_rules/`, `partial_order2/` (and `phrase_boundaries/`,
  `hag_benchmarks/`), each with the subfolders of the code: `hpo/` (as in hag, one Optuna database per dataset with
  all its studies, `marron1_db.sqlite3`, `japanese_vowels_db.sqlite3`; `bachue/hpo/studies.py`), `performances/` (the
  result CSVs of the final evaluations) and `analysis/` (paired comparisons, diagnostics, figures).

## Environment

Python of the environment `hag_env` (hag and reservoirpy installed in editable mode), from the root of the repository:

```bash
PYTHONPATH=$PWD python -m pytest tests -q
```

The variables of the environment keep their names of canopy (`HAG_CANARY_*`).
