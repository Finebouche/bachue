"""
Learning rules of the readout of the next-phrase prediction (bachue.tasks.next_phrase), compared on the
same reservoirs: from the least to the most biologically plausible,
  - "ridge": ridge regression on all the training rows at once (reservoirpy's Ridge, solver RIDGE_SOLVER), the
    reference of the other tasks;
  - "rls": online recursive least squares (bachue.models.readout_rules.rls: reservoirpy's RLS on the inputs and a
    constant input 1 for the bias, the RLS-based FORCE rule of Sussillo & Abbott 2009, non-local: each synapse uses
    the inverse correlation matrix of all the inputs), initial P = I / alpha; after one epoch, the weights of the
    ridge regression of coefficient alpha (the bias of reservoirpy's node, a running mean of the errors, used until
    2026-11-19, gave much worse readouts with a small alpha);
  - "lms": online least mean squares (the rule of reservoirpy's LMS, the LMS-based FORCE rule: local, but each readout
    needs its exact error; trained in JAX, bachue.models.readout_rules.lms), learning rate c / mean ||x||^2 (c the normalized rate,
    stable below 2): LMS of Widrow & Hoff, a constant learning rate (the mean computed once on the training rows);
  - "eh": the exploratory Hebbian rule of Hoerzer et al. (2014) (the rule of bachue.models.readout_rules.eh_numpy.EH, the paper's: filtered
    baselines, a single global binary reward for all the readouts, no teacher; trained in JAX, bachue.models.readout_rules.eh),
    learning rate c / mean ||x||^2;
  - "eh_np": the same with the baseline "sample" (node perturbation of Fiete & Seung 2006: the reward compared with
    that of the output without noise on the same input; not the paper's rule, which does not know the noise), global
    reward; "eh_np_local": with a reward for each readout (its own modulatory signal);
  - "nlms" and "neh": LMS and eh_np_local with the learning rate of each row normalized by its squared norm (NLMS,
    learning rate c / (||x||^2 + 1)), needed on the monomials (order 2) whose norms spread much more than those of
    the states. "nlms" is NLMS (Nagumo & Noda 1967, Albert & Gardner 1967; as a reservoir readout: Yuenyong 2014, and
    the step of BPDC, Steil 2004); "eh_np", "eh_np_local" and "neh" are ours (closest published rule of neh: the
    decorrelated rule of Burms, Caluwaerts & Dambre 2015). Which version is which: bachue.models.readout_rules.
Each rule can read the monomials of the state up to degree n instead of the state ("<rule>:<n>", e.g. "nlms:2"; the
features of the generalized readout of Ohkubo & Inubushi 2024): "ridge:n" the generalized readout
(bachue.models.generalized_readout), "nlms:n" and "neh:n" trained in their dual form on the kernel of the monomials
(bachue.models.readout_rules.kernel, the same weights as on the monomials themselves).
Each rule can also read the state and selected products of degree 2 ("<rule>:<selection>", a selection of
bachue.performances.partial_order2_evaluation chosen on the training rows before the training of the readout, e.g.
"nlms:pca:64": the 2080 products of the 64 first principal components; "eh_np_local:scores": the 435 products of the
outputs of the linear ridge readout; "ridge:random_pairs:2080"): inputs standardized by the selection (the state and
each product), in primal form; the linear readout of the selections "scores", "weights_high", "weights_low" and "residual" with its ridge
coefficient chosen on the training rows (Selection.chosen_ridge; nested for the out-of-fold outputs of the training
rows of "scores" and "residual": chosen without the fold of each row).
The online rules see the training rows song by song, each song in its time order, the songs in a random order at each
epoch (pass over the training songs). Their inputs are centered ("center", RLS only) or standardized ("standardize")
with the mean and standard deviation of each unit on the training rows (a homeostatic estimate of the mean activity and of the
gain of each input, local to its synapses): the states have a large common mean and small fluctuations (mean 0.16,
standard deviation 0.01 per unit for random_ee 500), and without this the online rules learn the direction of the
mean (accuracy at the level of chance). The paper's EH rule needs slowly changing inputs (its filtered baseline
estimates the noise from the change of the output): on the independent prediction points, it does not learn.

Reservoirs and readout of each condition: those of the studies of the readout (HAG_CANARY_TAIL_HPO=1, with the
variables of the studies: HAG_CANARY_CALLS, HAG_CANARY_LEAK), with the readout (training rows, decision, tail) of the
condition in their evaluation (column hpo of outputs/<study>/performances/<prefix>hpo_readouts.csv, e.g.
next_phrase_calls_tail_hpo_readouts.csv), or (HAG_CANARY_TAIL_HPO=0) those of the studies of the reservoirs with the
linear ridge readout, the readout "last" -> "last". The reservoirs are not optimized again for the rules: all the
rules read the same reservoirs (fold k of the cross-validation: drawn with the seed CV_SEED + k; seed s of the
evaluation: drawn with the seed s), and only the hyperparameters of each rule (ridge coefficient, alpha, normalized
learning rate, noise, number of epochs) are chosen, in the 3-fold cross-validation by song of the studies on the train
songs (out-of-fold accuracy; never on the test songs); then each rule is trained on all the train songs and scored on
the test songs for each seed (paired comparisons with "ridge"). Rules: HAG_CANARY_READOUT_RULES (comma-separated,
default all the rules on the states), their configurations trained in HAG_CANARY_RULES_JOBS processes (default 1),
for at most HAG_CANARY_RULES_MAX_EPOCHS epochs (default the last of EPOCHS).
These comparisons read reservoirs optimized for the ridge readout: they favor it (CLAUDE.md, fair comparisons). For a
fair comparison, each rule with its own studies: HAG_CANARY_HPO_RULE=<rule> (bachue.hpo.learning_rules_study, partial_order2_study,
reservoirs optimized for the rule, slurm/submit_hpo_rules.sh), then its evaluation here with the same variable: the
reservoirs of its studies, that rule alone, its rate, noise and alpha of the study (the hybrids, without study: those
of the grid), its number of epochs (and normalization of RLS, coefficient of ridge:<n>) chosen by the cross-validation.

Results: outputs/<study>/performances/<prefix>readout_rules<suffix>.csv (one row per condition, size, seed and rule),
<prefix>readout_rules<suffix>_comparisons.csv (each rule - ridge, paired by seed) and <prefix>readout_rules<suffix>_cv.csv
(out-of-fold accuracy of each configuration and number of epochs: the learning curves), suffix
HAG_CANARY_RULES_SUFFIX (default none).
Usage (HAG_CANARY_TASK=next_phrase HAG_CANARY_TAIL_HPO=1, with the variables of the studies):
  python -m bachue.performances.learning_rules_evaluation run <condition> <size> [nb_seeds]   one condition
                                                       (slurm/submit_readout_rules.sh), results in <name>_parts
  python -m bachue.performances.learning_rules_evaluation merge          results of the parts and the comparisons
  python -m bachue.performances.learning_rules_evaluation list <sizes>   "<condition> <size>" of the conditions (no _proj)
"""
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from bachue.analysis.comparisons import compare_rules
from bachue.common import OUTPUTS_DIR, load_dataset, pretrain_phrases
from bachue.hpo.learning_rules_study import JOBS, MAX_EPOCHS, choose, frame_songs_of
from bachue.models.readout_rules.training import normalized, rule_inputs, rule_order, selected_rows, trained
from bachue.models.reservoirs import build
from bachue.performances.readout_window_evaluation import NAME, all_conditions
from bachue.settings import HPO_RULE, SELECTION, STUDY_PREFIX, TAIL_HPO, TAIL_SEPARATE
from bachue.tasks.next_phrase import BASE_READOUT, metrics, prepare, readout_of, song_data

KIND = "partial_order2" if SELECTION else "learning_rules"  # (the kind of the studies of the rule)
OUTPUTS = OUTPUTS_DIR / KIND / "performances"
ANALYSIS = OUTPUTS_DIR / KIND / "analysis"  # (the paired comparisons)
READOUT_WINDOW_OUTPUTS = OUTPUTS_DIR / "readout_window" / "performances"  # (the readouts of the studies of HAG_CANARY_TAIL_HPO)

RULES = os.environ.get("HAG_CANARY_READOUT_RULES", "ridge,rls,lms,eh,eh_np,eh_np_local").split(",")
if HPO_RULE:  # (the studies of the rule, bachue.hpo.learning_rules_study: that rule alone)
    RULES = [HPO_RULE]
RESULTS = f"{STUDY_PREFIX}readout_rules{os.environ.get('HAG_CANARY_RULES_SUFFIX', '')}"
PARTS = OUTPUTS / f"{RESULTS}_parts"


def seed_scores(label, hyper, all_data, all_states, train_songs, test_songs, true, classes, seed, moments=None):
    """Metrics on the test songs of the rule label with its chosen hyperparameters hyper, trained on the train songs
    (the reservoirs of the seed)."""
    rule, order, selection = rule_inputs(label)
    config = {key: value for key, value in hyper.items() if key != "epochs"}
    if selection:
        data, inputs = selected_rows(selection, all_data, train_songs, moments, seed)
        X_test = inputs(np.vstack([all_states[w] for w in test_songs]))
    else:
        data, states = normalized(config, all_data, all_states, train_songs)
        X_test = np.vstack([states[w] for w in test_songs])
    epochs = [hyper["epochs"]] if hyper["epochs"] is not None else [None]
    _, node = next(trained(rule, config, data, train_songs, np.random.default_rng([seed, 1]), epochs, order,
                           bool(selection)))
    with np.errstate(all="ignore"):
        return metrics(np.nan_to_num(node.run(X_test), nan=0.0), true, classes)


def run(todo, nb_seeds):
    """(results of the rules (RULES) for each condition and seed, their learning curves in the cross-validation)."""
    songs, train_songs, test_songs, frame = load_dataset(top_db=None)
    classes = prepare(songs, train_songs, frame)
    studies = pd.read_csv(READOUT_WINDOW_OUTPUTS / f"{NAME}.csv") if TAIL_HPO else None
    rows, curves = [], []
    for condition, model, params, _, size in todo:
        if TAIL_HPO:  # (the readout of the study of the readout of the condition)
            study = studies[(studies.condition == condition) & (studies.network_size == size) & studies.hpo.astype(bool)]
            if study.empty:
                raise ValueError(f"{condition} ({size}): no readout in {READOUT_WINDOW_OUTPUTS / f'{NAME}.csv'}")
            points, decision, tail = study.iloc[0][["points", "decision", "tail"]]
            readout = readout_of(points, decision, None if pd.isna(tail) else int(tail))
        else:  # (the studies of the reservoirs with the linear ridge readout)
            readout = BASE_READOUT
            points, decision, tail = readout[:3]
        print(f"{condition} ({size}): readout {readout}", flush=True)
        # (the studies of a rule, HPO_RULE: its rate, noise and alpha of the study, the hybrids those of the grid)
        chosen, curve = choose(model, params, size, songs, train_songs, frame, classes, readout, RULES,
                               params.get("readout:rate"), MAX_EPOCHS, params.get("readout:noise"),
                               params.get("readout:alpha"))
        curves.append(curve.assign(condition=condition, network_size=size))
        for rule, (hyper, cv_accuracy) in chosen.items():
            print(f"  {rule}: {hyper}, cross-validation accuracy {100 * cv_accuracy:.2f} %", flush=True)
        true = np.concatenate([songs[w]["next"] for w in test_songs])
        seed_data = {}  # (all the rules on the same reservoirs, drawn with the seed)
        for seed in range(nb_seeds):
            np.random.seed(seed)
            reservoirs = build(model, params, size, pretrain_phrases(songs, train_songs, frame))
            all_states, all_data, moments = song_data(reservoirs, songs, train_songs + test_songs, readout, classes,
                                                      frame_songs_of(RULES, train_songs))
            seed_data[seed] = (all_data, all_states, moments)
        tasks = [(seed, rule) for seed in range(nb_seeds) for rule in chosen]
        results = Parallel(n_jobs=min(JOBS, len(tasks)))(
            delayed(seed_scores)(rule, chosen[rule][0], *seed_data[seed][:2], train_songs, test_songs, true, classes,
                                 seed, seed_data[seed][2])
            for seed, rule in tasks)
        for (seed, rule), result in zip(tasks, results):
            hyper, cv_accuracy = chosen[rule]
            rows.append(dict(condition=condition, network_size=size, seed=seed, rule=rule, order=rule_order(rule)[1],
                             selection=rule_inputs(rule)[2] or "", points=points, decision=decision, tail=tail, **hyper, cv_accuracy=cv_accuracy, **result))
        for seed in range(nb_seeds):
            print(f"  seed {seed}: " + ", ".join(f"{r['rule']} {100 * r['accuracy']:.1f} %" for r in rows
                                                 if r["seed"] == seed and r["condition"] == condition
                                                 and r["network_size"] == size), flush=True)
    return pd.DataFrame(rows), pd.concat(curves, ignore_index=True)


if __name__ == "__main__":
    if TAIL_SEPARATE:
        raise RuntimeError("HAG_CANARY_TAIL_HPO=0 (studies of the reservoirs) or 1 (studies of the readout) is needed")
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "list":
        for condition, _, _, _, size in all_conditions([int(s) for s in sys.argv[2].split(",")]):
            if not condition.endswith("_proj"):
                print(condition, size)
    elif command == "run":
        condition, size = sys.argv[2], int(sys.argv[3])
        nb_seeds = int(sys.argv[4]) if len(sys.argv) > 4 else 10
        todo = [c for c in all_conditions([size] if size else []) if c[0] == condition and c[4] == size]
        if not todo:
            raise ValueError(f"Unknown condition {condition} ({size})")
        PARTS.mkdir(parents=True, exist_ok=True)
        results, curves = run(todo, nb_seeds)
        results.to_csv(PARTS / f"{condition}_{size}.csv", index=False)
        curves.to_csv(PARTS / f"{condition}_{size}_cv.csv", index=False)
    elif command == "merge":
        parts = sorted(part for part in PARTS.glob("*.csv") if not part.stem.endswith("_cv"))
        print(f"{len(parts)} conditions")
        results = pd.concat([pd.read_csv(part) for part in parts], ignore_index=True)
        results.to_csv(OUTPUTS / f"{RESULTS}.csv", index=False)
        curve_parts = sorted(PARTS.glob("*_cv.csv"))
        if curve_parts:
            pd.concat([pd.read_csv(part) for part in curve_parts], ignore_index=True).to_csv(
                OUTPUTS / f"{RESULTS}_cv.csv", index=False)
        comparisons = compare_rules(results)
        pd.set_option("display.width", 200)
        print(comparisons.round(4).to_string(index=False))
        ANALYSIS.mkdir(parents=True, exist_ok=True)
        comparisons.to_csv(ANALYSIS / f"{RESULTS}_comparisons.csv", index=False)
        print("Saved to", OUTPUTS / f"{RESULTS}.csv", "and", ANALYSIS / f"{RESULTS}_comparisons.csv")
    else:
        raise ValueError(f"Unknown command {command!r}: run, merge or list")
