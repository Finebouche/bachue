"""
Final evaluations of the readouts on selected products of degree 2 (bachue.models.product_selection: the methods): can
a few hundred or thousand products, chosen before the regression, replace all those of the generalized readout?

The studies of bachue.hpo.partial_order2_study (the command "studies"): for each condition (the reservoirs of the
studies of BACHUE_STUDY_RULE=<rule>:<selection>, optimized for it), its rate, noise and alpha of the study, its number
of epochs (and ridge coefficient) chosen again by the 3-fold cross-validation on the train songs, then the rule
trained on all the train songs and scored on the test songs for each seed (reservoirs drawn with the seed). Results:
outputs/expansions/performances/<prefix>readout_rules.csv (one row per condition, size and seed) and _cv.csv,
the paired comparisons with the other conditions in outputs/expansions/analysis (bachue.analysis.comparisons).
Usage (BACHUE_TASK=next_phrase BACHUE_WITH_CALLS=1 BACHUE_SEARCH_READOUT=0 BACHUE_STUDY_RULE=<rule>:<selection>
[BACHUE_LEAKY_RESERVOIRS=1]; on a classification dataset, BACHUE_DATASET=<dataset>, also the references
BACHUE_STUDY_RULE=<rule> and <rule>:<order>, e.g. "nlms" and "ridge:2"):
  python -m bachue.performances.partial_order2_evaluation studies run <condition> <size> [nb_seeds]   (results in _parts)
  python -m bachue.performances.partial_order2_evaluation studies merge

The exploratory comparisons of the selections (the commands "run", "merge", "list" and "rules"). Exploratory: all the readouts read the same reservoirs, those of the studies of the variables of the environment
(BACHUE_STUDY_RULE=ridge:2: the reservoirs optimized for the full readout of order 2, which they approximate), not
reservoirs optimized for each (for that: BACHUE_STUDY_RULE=<rule>:<selection>, bachue.hpo.partial_order2_study).

Protocol: the ridge coefficient of each method chosen by the 3-fold cross-validation by song on the train songs
(choose), then each method trained on all the train songs and scored on the test songs for each
seed (reservoirs drawn with the seed). Diagnostics of seed 0: bachue.analysis.selection_diagnostics.

Results: outputs/<study>/performances/<prefix>monomial_selection.csv (one row per condition, size, seed and method),
<prefix>monomial_selection_cv.csv (out-of-fold accuracy of each method and coefficient) and
<prefix>monomial_selection_diagnostics.csv.
Usage (BACHUE_TASK=next_phrase BACHUE_STUDY_RULE=ridge:2, with the variables of the studies):
  python -m bachue.performances.partial_order2_evaluation run <condition> <size> [nb_seeds]   (results in _parts)
  python -m bachue.performances.partial_order2_evaluation merge
  python -m bachue.performances.partial_order2_evaluation list <sizes>      "<condition> <size>" of the pure models

The online rules on the same selected products (the command "rules"): the rules of
bachue.performances.learning_rules_evaluation instead of the ridge regression, on the same inputs. With few products,
the rules learn their weights themselves (primal form), without the kernel of the training rows of "nlms:2" and
"neh:2", and RLS, whose matrix has the size of the inputs, becomes possible. Exploratory as well: the same reservoirs.
  - rules: BACHUE_SELECTION_RULES (default nlms, eh_np_local, rls);
  - inputs (BACHUE_SELECTION_METHODS): "linear" (z alone), "full_std" (all the products of z, in dual form: only
    nlms and neh), or a selection ("weights_high:m", "residual:M", "pca:m", "random_pairs:M", "scores", ...): z and the
    selected products, in primal form;
  - protocol: the hyperparameters of each rule and method (the grids of bachue.hpo.learning_rules_study; noise of the
    versions of EH: NEH_NOISES; alpha of RLS: RLS_ALPHAS; number of epochs, at most BACHUE_ONLINE_MAX_EPOCHS, default
    1000) chosen by the 3-fold cross-validation by song on the train songs, then each rule and method trained on all
    the train songs and scored on the test songs for each seed.
Results: outputs/<study>/performances/<prefix>monomial_selection_rules.csv (one row per condition, size, seed, rule and method, with
the seconds of its training) and <prefix>monomial_selection_rules_cv.csv (out-of-fold accuracy of each configuration
and number of epochs).
Usage (the same variables):
  python -m bachue.performances.partial_order2_evaluation rules run <condition> <size> [nb_seeds]   (results in _parts)
  python -m bachue.performances.partial_order2_evaluation rules merge

The final evaluation of the studies of bachue.hpo.partial_order2_study (a rule and a selection, reservoirs optimized
for them): bachue.performances.learning_rules_evaluation with BACHUE_STUDY_RULE=<rule>:<selection>.
"""
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import KFold

from bachue.analysis.comparisons import compare_rules
from bachue.analysis.selection_diagnostics import selection_diagnostics
from bachue.common import OUTPUTS_DIR, RIDGES, load_dataset, pretrain_phrases
from bachue.hpo.learning_rules_study import JOBS as RULE_JOBS, MAX_EPOCHS as STUDY_MAX_EPOCHS, NEH_NOISES, choose as choose_rule, configurations
from bachue.hpo.studies import CV_SEED, FOLD_SEED
from bachue.models.product_selection import COMPONENT_KINDS, RIDGES_STD, SIZES, Selection, UNIT_KINDS
from bachue.models.readout_rules.training import rule_inputs as label_inputs, selected_rows, trained
from bachue.models.reservoirs import build
from bachue.performances.readout_window_evaluation import all_conditions
from bachue.settings import DATASET, DATASET_NAME, HPO_RULE, N_FOLDS, SELECTION, SONGS, STUDY_PREFIX, env
from bachue.tasks.next_phrase import BASE_READOUT, metrics, prepare, reservoir_data, song_data, stacked


# (one folder per dataset, e.g. outputs/expansions/performances/japanese_vowels/)
OUTPUTS = OUTPUTS_DIR / "expansions" / "performances" / DATASET_NAME
ANALYSIS = OUTPUTS_DIR / "expansions" / "analysis" / DATASET_NAME  # (the diagnostics of the selections)

# ---- Ridge coefficient of the readouts on selected products (bachue.models.product_selection), chosen by the 3-fold
# cross-validation by song on the train songs, all the methods on the same reservoirs: the first step of
# bachue.performances.partial_order2_evaluation (which describes the protocol).
JOBS = int(env("BACHUE_RULE_PROCESSES", "1"))  # processes (folds, then seeds)


def ridges_of(method):
    return RIDGES if method == "full" else RIDGES_STD


def correct_counts(selection, methods, X_val, true, classes):
    """{method: correct predictions on the rows X_val for each coefficient of ridges_of(method)}."""
    counts = {}
    for method in methods:
        readout, inputs, _ = selection.readout(method)
        F = inputs(X_val)
        counts[method] = np.array([np.sum(classes[readout.set_ridge(ridge).run(F).argmax(axis=1)] == true)
                                   for ridge in ridges_of(method)])
    return counts


def fold_counts(data, fit, val, true, classes, rho_linear, methods, fold):
    X, Y, moments = data
    selection = Selection(*stacked(X, Y, fit), moments, rho_linear, seed=CV_SEED + fold)
    return correct_counts(selection, methods, np.vstack([X[w] for w in val]), true, classes)


def choose(model, params, size, songs, train_songs, frame, classes, methods):
    """({method: (ridge coefficient, out-of-fold accuracy)}, out-of-fold accuracy of each method and coefficient,
    coefficient of the linear readout of the selections), in the 3-fold cross-validation by song of the studies."""
    folds = []
    for k, (fit_index, val_index) in enumerate(KFold(N_FOLDS, shuffle=True, random_state=FOLD_SEED).split(train_songs)):
        np.random.seed(CV_SEED + k)
        fit, val = [train_songs[i] for i in fit_index], [train_songs[i] for i in val_index]
        reservoirs = build(model, params, size, pretrain_phrases(songs, fit, frame))
        data = reservoir_data(reservoirs, songs, fit + val, classes, fit)
        folds.append((data, fit, val, np.concatenate([songs[w]["next"] for w in val])))
    total = sum(len(true) for *_, true in folds)
    linear = sum(fold_counts(data, fit, val, true, classes, 1.0, ["linear"], k)["linear"]
                 for k, (data, fit, val, true) in enumerate(folds))
    rho_linear = float(RIDGES_STD[linear.argmax()])
    counts = Parallel(n_jobs=min(JOBS, len(folds)))(
        delayed(fold_counts)(data, fit, val, true, classes, rho_linear, methods, k)
        for k, (data, fit, val, true) in enumerate(folds))
    correct = {method: sum(count[method] for count in counts) for method in methods}
    chosen = {method: (float(ridges_of(method)[c.argmax()]), c.max() / total) for method, c in correct.items()}
    curves = pd.DataFrame([dict(method=method, ridge=ridge, cv_accuracy=count / total)
                           for method, c in correct.items() for ridge, count in zip(ridges_of(method), c)])
    return chosen, curves, rho_linear


# ---- monomial selection
DEFAULT_METHODS = (["linear", "full", "full_std", "scores"]
                   + [f"{kind}:{m}" for kind in UNIT_KINDS + COMPONENT_KINDS for m in SIZES]
                   + [f"{kind}:{m * (m + 1) // 2}" for kind in ("residual", "random_pairs") for m in SIZES])
METHODS = env("BACHUE_SELECTION_METHODS", ",".join(DEFAULT_METHODS)).split(",")
RESULTS = f"{STUDY_PREFIX}monomial_selection{env("BACHUE_SELECTION_SUFFIX", '')}"
PARTS = OUTPUTS / f"{RESULTS}_parts"


def seed_results(data, train_songs, test_songs, true, classes, rho_linear, chosen, seed, diagnose):
    """([dict(method, inputs, metrics on the test songs)], diagnostics) of the methods trained on the train songs."""
    X, Y, moments = data
    selection = Selection(*stacked(X, Y, train_songs), moments, rho_linear, seed=seed)
    X_test = np.vstack([X[w] for w in test_songs])
    rows, diagnostics = [], []
    for method, (ridge, _) in chosen.items():
        readout, inputs, n_inputs = selection.readout(method)
        scores = readout.set_ridge(ridge).run(inputs(X_test))
        rows.append(dict(method=method, inputs=n_inputs, **metrics(scores, true, classes)))
        if diagnose and method == "full_std" and hasattr(readout, "dual_coef"):
            diagnostics = selection_diagnostics(selection, readout, list(chosen))
    return rows, diagnostics


def run(todo, nb_seeds, methods=METHODS):
    """(results of the methods for each condition and seed, their cross-validation, the diagnostics of seed 0)."""
    songs, train_songs, test_songs, frame = load_dataset(top_db=None)
    classes = prepare(songs, train_songs, frame)
    results, curves, diagnostics = [], [], []
    for condition, model, params, _, size in todo:
        keys = dict(condition=condition, network_size=size)
        chosen, curve, rho_linear = choose(model, params, size, songs, train_songs, frame, classes, methods)
        curves.append(curve.assign(**keys))
        print(f"{condition} ({size}): linear readout of the selections {rho_linear:.0e}", flush=True)
        for method, (ridge, accuracy) in chosen.items():
            print(f"  {method}: ridge {ridge:.0e}, cross-validation accuracy {100 * accuracy:.2f} %", flush=True)
        true = np.concatenate([songs[w]["next"] for w in test_songs])
        seed_data = []
        for seed in range(nb_seeds):
            np.random.seed(seed)
            reservoirs = build(model, params, size, pretrain_phrases(songs, train_songs, frame))
            seed_data.append(reservoir_data(reservoirs, songs, train_songs + test_songs, classes, train_songs))
        outputs = Parallel(n_jobs=min(JOBS, nb_seeds))(
            delayed(seed_results)(seed_data[seed], train_songs, test_songs, true, classes, rho_linear, chosen, seed,
                                  seed == 0)
            for seed in range(nb_seeds))
        for seed, (rows, diagnostic) in enumerate(outputs):
            results += [dict(**keys, seed=seed, **row, ridge=chosen[row["method"]][0],
                             cv_accuracy=chosen[row["method"]][1]) for row in rows]
            diagnostics += [dict(**keys, seed=seed, **row) for row in diagnostic]
            print(f"  seed {seed}: " + ", ".join(f"{row['method']} {100 * row['accuracy']:.1f} %" for row in rows),
                  flush=True)
    return pd.DataFrame(results), pd.concat(curves, ignore_index=True), pd.DataFrame(diagnostics)


# ---- The online rules on the selected products (the command "rules")
# ---- Hyperparameters of the online rules on the inputs of the selections of products, chosen by the 3-fold
# cross-validation by song on the train songs: the first step of bachue.performances.partial_order2_evaluation (which
# describes the rules, the methods and the protocol).
RULES = env("BACHUE_SELECTION_RULES", "nlms,eh_np_local,rls").split(",")
RULES_METHODS = env("BACHUE_SELECTION_METHODS", "linear,full_std,scores,weights_high:64,residual:2080,pca:64,"
                         "pca_frames:64,random_pairs:2080").split(",")
RLS_ALPHAS = [1e-8, 1e-6, 1e-4, 1e-2, 1.0]  # (initial P = I / alpha, on the standardized inputs)
RULES_MAX_EPOCHS = int(env("BACHUE_ONLINE_MAX_EPOCHS", "1000"))


def rule_configurations(rule):
    """[(hyperparameters, epochs)] of a rule (the inputs are already standardized)."""
    if rule == "rls":
        return [(dict(alpha=alpha), [1]) for alpha in RLS_ALPHAS]
    noises = [None] if rule in ("lms", "nlms", "neh") else NEH_NOISES  # (neh: its own grid of noises, the same)
    return [({key: value for key, value in hyper.items() if key != "normalization"}, epochs)
            for noise in noises for hyper, epochs in configurations(rule, max_epochs=RULES_MAX_EPOCHS, noise=noise)]


def pairs(methods=RULES_METHODS, rules=RULES):
    """(method, rule) trained: all the products (dual form, always normalized) for nlms and neh only."""
    return [(method, rule) for method in methods for rule in rules if method != "full_std" or rule in ("nlms", "neh")]


def rule_inputs(selection, method):
    """(inputs of the training rows, function of the states giving the inputs, degree of the monomials of the inputs
    read by the rule: 2 for all the products in dual form)."""
    if method in ("linear", "full_std"):
        return selection.Z, selection.standardized, 2 if method == "full_std" else 1
    F, inputs = selection.features(method)
    return F, inputs, 1


def trained_readouts(selection, waves, lengths, method, rule, hyper, epochs, rng):
    """Yields (epochs, readout, function of the states giving its inputs, number of inputs) of the rule trained on the
    inputs of the method of the training rows (those of the songs waves, lengths rows each), as readout_rules.trained."""
    F, inputs, order = rule_inputs(selection, method)
    bounds = np.cumsum([0] + list(lengths))
    rows = {w: (F[bounds[i]:bounds[i + 1]], selection.Y[bounds[i]:bounds[i + 1]]) for i, w in enumerate(waves)}
    n = F.shape[1]
    for epoch, readout in trained(rule, hyper, rows, list(waves), rng, epochs, order):
        yield epoch, readout, inputs, n + n * (n + 1) // 2 if order == 2 else n


def timed(function, *args):
    """(result, seconds) of function(*args)."""
    start = time.time()
    return function(*args), time.time() - start


def configuration_counts(data, fit, val, true, classes, rho_linear, method, rule, c, hyper, epochs, fold):
    """{epochs: correct predictions on the songs val} of a configuration trained on the songs fit."""
    X, Y, moments = data
    selection = Selection(*stacked(X, Y, fit), moments, rho_linear, seed=CV_SEED + fold)
    X_val, F_val, counts = np.vstack([X[w] for w in val]), None, {}
    rng = np.random.default_rng([CV_SEED + fold, c])
    with np.errstate(all="ignore"):
        for epoch, readout, inputs, _ in trained_readouts(selection, fit, [len(X[w]) for w in fit], method, rule, hyper,
                                                          epochs, rng):
            if F_val is None:
                F_val = inputs(X_val)
            scores = np.nan_to_num(np.asarray(readout.run(F_val)), nan=-np.inf)
            counts[epoch] = int(np.sum(classes[scores.argmax(axis=1)] == true))
    return counts


def choose_rules(model, params, size, songs, train_songs, frame, classes, methods=RULES_METHODS, rules=RULES):
    """({(method, rule): (hyperparameters with "epochs", out-of-fold accuracy, seconds of all its configurations)},
    out-of-fold accuracy of each configuration and number of epochs, coefficient of the linear readout of the
    selections), in the 3-fold cross-validation by song of the studies."""
    folds = []
    for k, (fit_index, val_index) in enumerate(KFold(N_FOLDS, shuffle=True, random_state=FOLD_SEED).split(train_songs)):
        np.random.seed(CV_SEED + k)
        fit, val = [train_songs[i] for i in fit_index], [train_songs[i] for i in val_index]
        reservoirs = build(model, params, size, pretrain_phrases(songs, fit, frame))
        data = reservoir_data(reservoirs, songs, fit + val, classes, fit)
        folds.append((data, fit, val, np.concatenate([songs[w]["next"] for w in val])))
    total = sum(len(true) for *_, true in folds)
    linear = sum(fold_counts(data, fit, val, true, classes, 1.0, ["linear"], k)["linear"]
                 for k, (data, fit, val, true) in enumerate(folds))
    rho_linear = float(RIDGES_STD[linear.argmax()])
    tasks = [(k, method, rule, c, hyper, epochs) for k in range(len(folds)) for method, rule in pairs(methods, rules)
             for c, (hyper, epochs) in enumerate(rule_configurations(rule))]
    counts = Parallel(n_jobs=min(JOBS, len(tasks)))(
        delayed(timed)(configuration_counts, *folds[k], classes, rho_linear, method, rule, c, hyper, epochs, k)
        for k, method, rule, c, hyper, epochs in tasks)
    correct, seconds = {}, {}
    for (k, method, rule, c, hyper, epochs), (count, duration) in zip(tasks, counts):
        seconds[(method, rule)] = seconds.get((method, rule), 0) + duration
        for epoch, value in count.items():
            correct[(method, rule, c, epoch)] = correct.get((method, rule, c, epoch), 0) + value
    chosen, curves = {}, []
    for method, rule in pairs(methods, rules):
        configs = rule_configurations(rule)
        (c, epoch), count = max((((c, epoch), count) for (m, r, c, epoch), count in correct.items()
                                 if (m, r) == (method, rule)), key=lambda item: item[1])
        chosen[(method, rule)] = (dict(configs[c][0], epochs=epoch), count / total, seconds[(method, rule)])
        curves += [dict(method=method, rule=rule, **configs[c][0], epochs=epoch, cv_accuracy=count / total)
                   for (m, r, c, epoch), count in correct.items() if (m, r) == (method, rule)]
    return chosen, pd.DataFrame(curves), rho_linear


# ---- monomial selection rules
RULES_RESULTS = f"{STUDY_PREFIX}monomial_selection_rules{env("BACHUE_SELECTION_SUFFIX", '')}"
RULES_PARTS = OUTPUTS / f"{RULES_RESULTS}_parts"


def seed_metrics(data, train_songs, test_songs, true, classes, rho_linear, method, rule, hyper, seed):
    """dict(inputs, metrics on the test songs) of the rule trained on the inputs of the method of the train songs."""
    X, Y, moments = data
    selection = Selection(*stacked(X, Y, train_songs), moments, rho_linear, seed=seed)
    config = {key: value for key, value in hyper.items() if key != "epochs"}
    with np.errstate(all="ignore"):
        _, readout, inputs, n_inputs = next(trained_readouts(
            selection, train_songs, [len(X[w]) for w in train_songs], method, rule, config, [hyper["epochs"]],
            np.random.default_rng([seed, 1])))
        scores = np.nan_to_num(np.asarray(readout.run(inputs(np.vstack([X[w] for w in test_songs])))), nan=0.0)
    return dict(inputs=n_inputs, **metrics(scores, true, classes))


def run_rules(todo, nb_seeds, methods=RULES_METHODS, rules=RULES):
    """(results of the rules on the methods for each condition and seed, their cross-validation)."""
    songs, train_songs, test_songs, frame = load_dataset(top_db=None)
    classes = prepare(songs, train_songs, frame)
    results, curves = [], []
    for condition, model, params, _, size in todo:
        keys = dict(condition=condition, network_size=size)
        chosen, curve, rho_linear = choose_rules(model, params, size, songs, train_songs, frame, classes, methods, rules)
        curves.append(curve.assign(**keys))
        print(f"{condition} ({size}): linear readout of the selections {rho_linear:.0e}", flush=True)
        for (method, rule), (hyper, accuracy, seconds) in chosen.items():
            print(f"  {rule} on {method}: {hyper}, cross-validation accuracy {100 * accuracy:.2f} %, {seconds:.0f} s",
                  flush=True)
        true = np.concatenate([songs[w]["next"] for w in test_songs])
        seed_data = []
        for seed in range(nb_seeds):
            np.random.seed(seed)
            reservoirs = build(model, params, size, pretrain_phrases(songs, train_songs, frame))
            seed_data.append(reservoir_data(reservoirs, songs, train_songs + test_songs, classes, train_songs))
        tasks = [(seed, method, rule) for seed in range(nb_seeds) for method, rule in chosen]
        outputs = Parallel(n_jobs=min(JOBS, len(tasks)))(
            delayed(timed)(seed_metrics, seed_data[seed], train_songs, test_songs, true, classes, rho_linear, method,
                           rule, chosen[(method, rule)][0], seed)
            for seed, method, rule in tasks)
        for (seed, method, rule), (output, seconds) in zip(tasks, outputs):
            hyper, accuracy, _ = chosen[(method, rule)]
            results.append(dict(**keys, seed=seed, rule=rule, method=method, **hyper, cv_accuracy=accuracy, **output,
                                seconds=seconds))
        for seed in range(nb_seeds):
            print(f"  seed {seed}: " + ", ".join(
                f"{r['rule']} on {r['method']} {100 * r['accuracy']:.1f} % ({r['seconds']:.0f} s)"
                for r in results[-len(tasks):] if r["seed"] == seed), flush=True)
    return pd.DataFrame(results), pd.concat(curves, ignore_index=True)


# ---- The final evaluation of the studies of bachue.hpo.partial_order2_study (the command "studies")
STUDIES_RESULTS = f"{STUDY_PREFIX}readout_rules{env("BACHUE_EVAL_RULES_SUFFIX", '')}"
STUDIES_PARTS = OUTPUTS / f"{STUDIES_RESULTS}_parts"


def study_seed_scores(label, hyper, all_data, all_states, train_songs, test_songs, true, classes, seed):
    """Metrics on the test songs of the rule and selection label with the hyperparameters hyper of its study, trained
    on the train songs (the reservoirs of the seed; the selection chosen on the train songs)."""
    rule, order, selection = label_inputs(label)
    config = {key: value for key, value in hyper.items() if key != "epochs"}
    data, inputs = selected_rows(selection, all_data, train_songs, None, seed)
    X_test = inputs(np.vstack([all_states[w] for w in test_songs]))
    epochs = [hyper["epochs"]] if hyper["epochs"] is not None else [None]
    _, node = next(trained(rule, config, data, train_songs, np.random.default_rng([seed, 1]), epochs, order, True))
    with np.errstate(all="ignore"):
        return metrics(np.nan_to_num(node.run(X_test), nan=0.0), true, classes)


def run_studies(todo, nb_seeds):
    """(results of the rule and selection HPO_RULE for each condition and seed, its cross-validation)."""
    songs, train_songs, test_songs, frame = load_dataset(top_db=None)
    classes = prepare(songs, train_songs, frame)
    rows, curves = [], []
    for condition, model, params, _, size in todo:
        print(f"{condition} ({size}): {HPO_RULE}", flush=True)
        # (the rate, noise and alpha of the study; the hybrids, without study: those of the grid)
        chosen, curve = choose_rule(model, params, size, songs, train_songs, frame, classes, BASE_READOUT, [HPO_RULE],
                                    params.get("readout:rate"), STUDY_MAX_EPOCHS, params.get("readout:noise"),
                                    params.get("readout:alpha"))
        curves.append(curve.assign(condition=condition, network_size=size))
        hyper, cv_accuracy = chosen[HPO_RULE]
        print(f"  {hyper}, cross-validation accuracy {100 * cv_accuracy:.2f} %", flush=True)
        true = np.concatenate([songs[w]["next"] for w in test_songs])
        seed_data = []
        for seed in range(nb_seeds):
            np.random.seed(seed)
            reservoirs = build(model, params, size, pretrain_phrases(songs, train_songs, frame))
            all_states, all_data, _ = song_data(reservoirs, songs, train_songs + test_songs, BASE_READOUT, classes)
            seed_data.append((all_data, all_states))
        results = Parallel(n_jobs=min(RULE_JOBS, nb_seeds))(
            delayed(study_seed_scores)(HPO_RULE, hyper, *seed_data[seed], train_songs, test_songs, true, classes, seed)
            for seed in range(nb_seeds))
        for seed, result in enumerate(results):
            rows.append(dict(condition=condition, network_size=size, seed=seed, rule=HPO_RULE, order=1,
                             selection=label_inputs(HPO_RULE)[2], points="last", decision="last", tail=None, **hyper,
                             cv_accuracy=cv_accuracy, **result))
            print(f"  seed {seed}: {100 * result['accuracy']:.1f} %", flush=True)
    return pd.DataFrame(rows), pd.concat(curves, ignore_index=True)


def run_studies_on_instances(todo, nb_seeds):
    """run_studies on a classification dataset of hag (BACHUE_DATASET): the last state of each instance, the
    hyperparameters of the rule chosen by bachue.hpo.learning_rules_study.choose_on_instances on the train instances
    (with the rate, noise and alpha of the study), then the rule trained on all the train instances and scored on the
    test instances for each seed (the protocol of bachue.performances.hag_benchmarks_evaluation)."""
    from bachue.datasets.benchmarks import instance_data, load
    from bachue.hpo.learning_rules_study import choose_on_instances
    from bachue.performances.hag_benchmarks_evaluation import seed_scores
    data = load(DATASET)
    true = np.asarray(data.Y_test).argmax(axis=1)
    rows, curves = [], []
    for condition, model, params, _, size in todo:
        print(f"{condition} ({size}): {HPO_RULE}", flush=True)
        chosen, curve = choose_on_instances(model, params, size, data, [HPO_RULE], params.get("readout:rate"),
                                            STUDY_MAX_EPOCHS, params.get("readout:noise"), params.get("readout:alpha"))
        curves.append(curve.assign(condition=condition, network_size=size))
        hyper, cv_accuracy = chosen[HPO_RULE]
        print(f"  {hyper}, cross-validation accuracy {100 * cv_accuracy:.2f} %", flush=True)
        seed_data = []
        for seed in range(nb_seeds):
            np.random.seed(seed)
            reservoirs = build(model, params, size, data.pretrain)
            train_states, train_rows = instance_data(reservoirs, data.train, data.Y_train)
            test_states, _ = instance_data(reservoirs, data.test, data.Y_test)
            seed_data.append((train_rows, train_states, test_states))
        results = Parallel(n_jobs=min(RULE_JOBS, nb_seeds))(
            delayed(seed_scores)(HPO_RULE, hyper, *seed_data[seed], true, seed) for seed in range(nb_seeds))
        for seed, result in enumerate(results):
            rows.append(dict(dataset=DATASET, condition=condition, network_size=size, seed=seed, rule=HPO_RULE,
                             order=1, selection=label_inputs(HPO_RULE)[2], **hyper, cv_accuracy=cv_accuracy, **result))
            print(f"  seed {seed}: {100 * result['accuracy']:.1f} %", flush=True)
    return pd.DataFrame(rows), pd.concat(curves, ignore_index=True)


def studies_command(arguments):
    """The command "studies": run <condition> <size> [nb_seeds], merge."""
    if not SELECTION and SONGS:  # (on a classification dataset: also the references, the rule on the states or on
        # all the monomials of an order, "nlms", "ridge:2"; on the songs, bachue.performances.learning_rules_evaluation)
        raise ValueError(f"BACHUE_STUDY_RULE={HPO_RULE}: a rule and a selection, <rule>:<selection>")
    command = arguments[0] if arguments else ""
    if command == "run":
        condition, size = arguments[1], int(arguments[2])
        nb_seeds = int(arguments[3]) if len(arguments) > 3 else 10
        todo = [c for c in all_conditions([size]) if c[0] == condition and c[4] == size]
        if not todo:
            raise ValueError(f"Unknown condition {condition} ({size})")
        STUDIES_PARTS.mkdir(parents=True, exist_ok=True)
        results, curves = (run_studies if SONGS else run_studies_on_instances)(todo, nb_seeds)
        results.to_csv(STUDIES_PARTS / f"{condition}_{size}.csv", index=False)
        curves.to_csv(STUDIES_PARTS / f"{condition}_{size}_cv.csv", index=False)
    elif command == "merge":
        parts = sorted(part for part in STUDIES_PARTS.glob("*.csv") if not part.stem.endswith("_cv"))
        print(f"{len(parts)} conditions")
        results = pd.concat([pd.read_csv(part) for part in parts], ignore_index=True)
        OUTPUTS.mkdir(parents=True, exist_ok=True)
        results.to_csv(OUTPUTS / f"{STUDIES_RESULTS}.csv", index=False)
        curve_parts = sorted(STUDIES_PARTS.glob("*_cv.csv"))
        if curve_parts:
            pd.concat([pd.read_csv(part) for part in curve_parts], ignore_index=True).to_csv(
                OUTPUTS / f"{STUDIES_RESULTS}_cv.csv", index=False)
        comparisons = compare_rules(results)
        ANALYSIS.mkdir(parents=True, exist_ok=True)
        comparisons.to_csv(ANALYSIS / f"{STUDIES_RESULTS}_comparisons.csv", index=False)
        print("Saved to", OUTPUTS / f"{STUDIES_RESULTS}.csv", "and", ANALYSIS / f"{STUDIES_RESULTS}_comparisons.csv")
    else:
        raise ValueError(f"Unknown command studies {command!r}: studies run or studies merge")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "studies":  # (the final evaluation of the studies)
        studies_command(sys.argv[2:])
        sys.exit()
    if len(sys.argv) > 1 and sys.argv[1] == "rules":  # (the online rules on the selected products)
        command = sys.argv[2] if len(sys.argv) > 2 else ""
        if command == "run":
            condition, size = sys.argv[3], int(sys.argv[4])
            nb_seeds = int(sys.argv[5]) if len(sys.argv) > 5 else 10
            todo = [c for c in all_conditions([size]) if c[0] == condition and c[4] == size]
            if not todo:
                raise ValueError(f"Unknown condition {condition} ({size})")
            RULES_PARTS.mkdir(parents=True, exist_ok=True)
            for table, suffix in zip(run_rules(todo, nb_seeds), ("", "_cv")):
                table.to_csv(RULES_PARTS / f"{condition}_{size}{suffix}.csv", index=False)
        elif command == "merge":
            for suffix in ("", "_cv"):
                parts = sorted(part for part in RULES_PARTS.glob("*.csv") if part.stem.endswith("_cv") == bool(suffix))
                if parts:
                    OUTPUTS.mkdir(parents=True, exist_ok=True)
                    pd.concat([pd.read_csv(part) for part in parts], ignore_index=True).to_csv(
                        OUTPUTS / f"{RULES_RESULTS}{suffix}.csv", index=False)
                    print(f"{len(parts)} conditions: {OUTPUTS / f'{RULES_RESULTS}{suffix}.csv'}")
        else:
            raise ValueError(f"Unknown command rules {command!r}: rules run or rules merge")
        sys.exit()
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "list":
        for condition, _, _, _, size in all_conditions([int(s) for s in sys.argv[2].split(",")]):
            if size and "+" not in condition and not condition.endswith("_proj"):
                print(condition, size)
    elif command == "run":
        condition, size = sys.argv[2], int(sys.argv[3])
        nb_seeds = int(sys.argv[4]) if len(sys.argv) > 4 else 10
        todo = [c for c in all_conditions([size]) if c[0] == condition and c[4] == size]
        if not todo:
            raise ValueError(f"Unknown condition {condition} ({size})")
        PARTS.mkdir(parents=True, exist_ok=True)
        for table, suffix in zip(run(todo, nb_seeds), ("", "_cv", "_diagnostics")):
            table.to_csv(PARTS / f"{condition}_{size}{suffix}.csv", index=False)
    elif command == "merge":
        for suffix in ("", "_cv", "_diagnostics"):
            parts = sorted(part for part in PARTS.glob("*.csv")
                           if part.stem.endswith(suffix) and (suffix or not part.stem.endswith(("_cv", "_diagnostics"))))
            if parts:
                folder = ANALYSIS if suffix == "_diagnostics" else OUTPUTS
                folder.mkdir(parents=True, exist_ok=True)
                pd.concat([pd.read_csv(part) for part in parts], ignore_index=True).to_csv(
                    folder / f"{RESULTS}{suffix}.csv", index=False)
                print(f"{len(parts)} conditions: {folder / f'{RESULTS}{suffix}.csv'}")
    else:
        raise ValueError(f"Unknown command {command!r}: run, merge or list")
