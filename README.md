# bachue

*Bioplausible Adaptation of Compact Higher-order Uniform Expansions.*

Learning rules and readouts of order 2 for reservoirs, studied on the prediction of the next phrase of canary songs.
The code was copied from [canopy](https://github.com/Finebouche/canopy) on 2026-11-19 (package `canopy` renamed
`bachue`); canopy keeps the other canary tasks (syllable classification, syllable generation, phrase certainty).

## Contents

The package follows the layout of [hag](https://github.com/Finebouche/hag): the hyperparameters, the final
evaluations and the analyses are separate.

- `bachue/datasets/`: the canary songs (`load_categorical_forecasting.py`) and the benchmarks of HAG's evaluations
  (`benchmarks.py`: `load("JapaneseVowels")`, ...), with the loaders and the preparation copied from `hag.datasets`
  (`load_*.py`, `pipeline.py`, `preprocessing.py`, `spectral_decomposition.py` with the MFCC of `mfcc_configs.py`;
  without `peak_centered_decomposition.py` and its representation "custom"), and three audio datasets of the wingbeats
  of insects from timeseriesclassification.com (`FruitFlies`, `InsectSound`, `MosquitoSound`). All the data are in
  `bachue/datasets/datasets/` (not tracked): downloaded at their first use (the audio of marron1 from Zenodo, the
  datasets of hag), except the corrected annotations of marron1, not published yet, to copy by hand into
  `bachue/datasets/datasets/marron1/annotations/` (from our computer to PlaFRIM, for instance).
- `bachue/models/`: the reservoirs, from hag (`reservoirs.py`); the readouts trained online, `readout_rules/` (LMS and
  NLMS, recursive least squares with the bias as the weight of a constant input, the exploratory Hebbian rule and its
  versions, their dual forms on the monomials, and `training.py`: a readout trained by a rule from its label
  `<rule>`, `<rule>:2` or `<rule>:<selection>`); the generalized readout of order 2 (`generalized_readout.py`); the
  selections of products of degree 2 (`product_selection.py`: random pairs, principal components, linear scores,
  published alternatives).
- `bachue/tasks/`: the prediction of the next phrase (`next_phrase.py`: prediction points, readouts, metrics), the
  phrase boundaries, and the classification of instances.
- `bachue/hpo/`: the hyperparameters. One file per study of the reservoirs (Optuna): `readout_window_study.py` (the readout:
  last, mean, tail, tail_mean, all; `BACHUE_STUDY_READOUT`), `learning_rules_study.py` (LMS, NLMS, RLS, the versions of EH,
  and all the monomials of order 2: `ridge:2`, `nlms:2`, `neh:2`; `BACHUE_STUDY_RULE`), `partial_order2_study.py`
  (a rule on selected products: `nlms:pca:64`, `ridge:random_pairs:2080`, ...), their shared parts in `studies.py`
  (search space of the reservoirs, run of a study). The other modules choose the hyperparameters of the readouts by
  cross-validation on the train set (`ridge.py`, `readout_rules.py`, `monomial_selection.py`, ...).
- `bachue/performances/`: the final evaluations on the test set, one file per study: `readout_window_evaluation.py`,
  `learning_rules_evaluation.py`, `partial_order2_evaluation.py` (its command `studies` for the studies of
  `partial_order2_study`, and the exploratory comparisons of the selections); `hag_benchmarks_evaluation.py` (the rules
  on the datasets of hag: `python -m bachue.performances.hag_benchmarks_evaluation JapaneseVowels random_ee 10`) and
  `phrase_boundaries_evaluation.py`.
- `bachue/analysis/`: the paired comparisons of the results, the diagnostics of the selections, and the figure of the
  studies of the selections (`studies_figure.py`: `visualisations/résultats/selections_hpo.html`, updated from
  the databases of the studies with `python -m bachue.analysis.studies_figure`).
- `bachue/settings.py`: the task, its variants and the names of the studies, from the variables of the environment.
- `slurm/`: job scripts for PlaFRIM (`submit_hpo_rules.sh`: the studies of the rules, `DATASETS="marron1 JapaneseVowels"`;
  `merge_studies.sh`: the parts of the jobs merged into the databases; `plafrim_usage.py`: a page of my use of
  PlaFRIM, run from my computer: nodes, core-hours of the day against the limit of the charter, pending jobs, quotas).
  `tests/`: unit tests.
- `outputs/`: one folder per kind of study, `readout_window/` (the ridge readout), `learning_rules/` (all the rules on the
  states: the 4 EH, the 2 LMS, RLS and ridge), `expansions/` (what the readout reads beyond the states: all the monomials
  of order 2, selected products, kernels of the inputs, with the best rules ridge, nlms and eh_np_local) (and `phrase_boundaries/`,
  `hag_benchmarks/`), each with the subfolders of the code: `hpo/` (as in hag, one Optuna database per dataset with
  all its studies, `marron1_db.sqlite3`, `japanese_vowels_db.sqlite3`; `bachue/hpo/studies.py`), `performances/` (the
  result CSVs of the final evaluations), `analysis/` (paired comparisons, diagnostics), both with one subfolder per dataset
  (`marron1/`, `japanese_vowels/`) and `figures/` (figures made by the code of a study).
- `visualisations/`: the HTML pages, `cours/` (explanations: the kernel trick, the selections of products) and
  `résultats/` (the results: the studies of the selections, the cost of computation, the rank of Φ), listed in
  `visualisations/index.html`.

## Environment

Python of the environment `hag_env` (hag and reservoirpy installed in editable mode), from the root of the repository:

```bash
PYTHONPATH=$PWD python -m pytest tests -q
```

The variables of the environment are named `BACHUE_*`. Their former names of canopy (`HAG_CANARY_*`) are still read
when the new one is not set (`bachue.settings.env` and `RENAMED` for Python, `slurm/env_names.sh` for the scripts), as long
as jobs submitted with them run.

| Variable | Former name | Meaning |
|---|---|---|
| `BACHUE_DATASET` | `HAG_CANARY_DATASET` | dataset: `marron1` (default) or a classification dataset of hag (`JapaneseVowels`) |
| `BACHUE_TASK` | `HAG_CANARY_TASK` | task on the songs: `boundaries` (default) or `next_phrase` |
| `BACHUE_LEAKY_RESERVOIRS` | `HAG_CANARY_LEAK` | `1`: leaky reservoirs, the leak rate in the search (studies `leak_`) |
| `BACHUE_WITH_CALLS` | `HAG_CANARY_CALLS` | `1`: the calls are phrases too (studies `next_phrase_calls_`) |
| `BACHUE_RESET_EACH_PHRASE` | `HAG_CANARY_RESET` | `1`: reservoirs reset at each phrase |
| `BACHUE_SEARCH_READOUT` | `HAG_CANARY_TAIL_HPO` | `1`: the readout window in the search, `2`: one tail for the training and one for the decision |
| `BACHUE_STUDY_RULE` | `HAG_CANARY_HPO_RULE` | rule the reservoirs are optimized for (`nlms`, `ridge:2`, `nlms:pca:64`) |
| `BACHUE_STUDY_READOUT` | `HAG_CANARY_HPO_READOUT` | fixed readout of the studies (`tail_mean:tail_mean:10`) |
| `BACHUE_STUDY_MAX_EPOCHS` | `HAG_CANARY_HPO_RULE_MAX_EPOCHS` | most epochs of an online rule in its studies (3000) |
| `BACHUE_KERNEL` | `HAG_CANARY_KERNEL` | kernel of the inputs, without reservoir (`volterra`) |
| `BACHUE_N_TRIALS` | `HAG_CANARY_N_TRIALS` | trials of a study (400) |
| `BACHUE_WORKERS` | `HAG_CANARY_WORKERS` | processes of a study (`auto`: from the cores and the memory) |
| `BACHUE_WORKER_MEM_GB` | `HAG_CANARY_WORKER_MEM` | memory of a process, GB (11) |
| `BACHUE_WORKER_MEM_GB_ALL_ROWS` | `HAG_CANARY_WORKER_MEM_ALL` | the same for the training rows `all` (28) |
| `BACHUE_JOB_STUDIES_DIR` | `HAG_CANARY_STUDIES` | directory of the database of a job (set by `slurm/canary.sbatch`) |
| `BACHUE_JOB_MODE` | `HAG_CANARY_MODE` | what `slurm/canary.sbatch` runs (`hpo`, `readout_rules`, ...) |
| `BACHUE_TASK_FILE` | `HAG_CANARY_TASKS` | file of the tasks of a job array (one line per task) |
| `BACHUE_EVAL_RULES` | `HAG_CANARY_READOUT_RULES` | rules compared by `learning_rules_evaluation` |
| `BACHUE_EVAL_RULES_SUFFIX` | `HAG_CANARY_RULES_SUFFIX` | suffix of the results of the rules |
| `BACHUE_ONLINE_MAX_EPOCHS` | `HAG_CANARY_RULES_MAX_EPOCHS` | most epochs of an online rule in the evaluations |
| `BACHUE_RULE_PROCESSES` | `HAG_CANARY_RULES_JOBS` | processes training the configurations of the rules |
| `BACHUE_BENCHMARK_RULES` | `HAG_CANARY_BENCHMARK_RULES` | rules of `hag_benchmarks_evaluation` |
| `BACHUE_SELECTION_METHODS` | `HAG_CANARY_SELECTION_METHODS` | selections of products compared |
| `BACHUE_SELECTION_RULES` | `HAG_CANARY_SELECTION_RULES` | rules on the selected products |
| `BACHUE_SELECTION_SIZES` | `HAG_CANARY_SELECTION_SIZES` | numbers of products of the sized selections (32,64,128) |
| `BACHUE_SELECTION_SUFFIX` | `HAG_CANARY_SELECTION_SUFFIX` | suffix of the results of the selections |
| `BACHUE_READOUT_ORDER` | `HAG_CANARY_READOUT_ORDER` | order of the ridge readout (1: linear, n: generalized) |
| `BACHUE_RIDGE_SOLVER` | `HAG_CANARY_RIDGE_SOLVER` | solver of reservoirpy's Ridge (`svd`) |
| `BACHUE_READOUT_TRAINING_ROWS` | `HAG_CANARY_POINTS` | training rows of the readout (`last`, `mean`, `all`, `tail`, `tail_mean`) |
| `BACHUE_READOUT_DECISION` | `HAG_CANARY_DECISION` | decision of the readout |
| `BACHUE_READOUT_TAIL` | `HAG_CANARY_TAIL` | frames of the tail (10) |
| `BACHUE_READOUT_DECISION_TAIL` | `HAG_CANARY_DECISION_TAIL` | frames of the tail of the decision |
