"""
Study 2: the learning rule of the readout "last" -> "last" (bachue.models.readout_rules, bachue.performances.learning_rules_evaluation
for their description): one study per rule, so that the rules are compared on reservoirs optimized for each of them
(results: bachue.performances.learning_rules_evaluation with the same variables).

Rule of the study: HAG_CANARY_HPO_RULE (studies "next_phrase_calls_[leak_]<rule>_<model>_<size>",
"<rule>-order2" for the rules on all the monomials of degree 2), on the states:
  - "lms" (LMS), "nlms" (normalized LMS), "rls" (recursive least squares);
  - the versions of the exploratory Hebbian rule: "eh" (Hoerzer et al. 2014), "eh_np", "eh_np_local", "neh";
  - or on the state and all its monomials of degree 2 (the generalized readout of Ohkubo & Inubushi 2024): "ridge:2"
    (ridge regression), "nlms:2" and "neh:2" (in dual form).
(The linear ridge readout, "ridge": the studies of "last" -> "last", bachue.hpo.readout_window_study. The rules on selected
products: bachue.hpo.partial_order2_study, with its own objective and the same ranges.)

Search: the parameters of the reservoirs (bachue.hpo.studies.reservoir_parameters, 400 trials by default) and those of
the rule: its normalized learning rate (LMS, NLMS and EH: "readout:rate", log-uniform in RULE_RATES; one training per
fold), the amplitude of the exploration noise of EH ("readout:noise", log-uniform in RULE_NOISES) and the coefficient
alpha of RLS (initial P = I / alpha, "readout:alpha", log-uniform in RULE_ALPHAS, one epoch). Objective: the
out-of-fold accuracy of the next phrase in the 3-fold cross-validation by song on the train songs; the other
hyperparameters of the rule (number of epochs, up to HPO_MAX_EPOCHS; normalization of the inputs of RLS; the ridge
coefficient of "ridge:2") are chosen in this cross-validation (bachue.hpo.learning_rules_study.choose; user attribute
"readout_rule" of the trial).

Usage (HAG_CANARY_TASK=next_phrase HAG_CANARY_CALLS=1 HAG_CANARY_TAIL_HPO=0 HAG_CANARY_HPO_RULE=<rule> [HAG_CANARY_LEAK=1]):
  python -m bachue.hpo.learning_rules_study <model> <size>     (slurm/submit_hpo_rules.sh: all the rules)
"""
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import KFold

from bachue.common import RIDGES, pretrain_phrases
from bachue.datasets.benchmarks import instance_data
from bachue.hpo.studies import CV_SEED, FOLD_SEED, reservoir_parameters, run_study
from bachue.models.product_selection import RIDGES_STD
from bachue.models.readout_rules.training import normalized, rule_inputs, rule_order, selected_rows, trained
from bachue.models.reservoirs import build
from bachue.settings import HPO_RULE, N_FOLDS
from bachue.tasks.next_phrase import BASE_READOUT, song_data


# ---- Hyperparameters of the learning rules of the readout (bachue.models.readout_rules.training), chosen by the
# 3-fold cross-validation by song on the train songs: the grids of each rule (configurations), the training and the
# scores of a configuration on a fold (configuration_counts) and the choice of the best configuration and number of
# epochs of each rule (choose). Used by the objective of the studies of a rule (bachue.hpo.learning_rules_study and
# partial_order2_study, HAG_CANARY_HPO_RULE) and by the final evaluation (bachue.performances.learning_rules_evaluation, which
# describes the rules and the protocol).
NORMALIZATIONS = ["center", "standardize"]  # (inputs of RLS; LMS and EH: "standardize", better in all the conditions)
RLS_ALPHAS = 10.0 ** np.arange(-10, -2)  # (ridge coefficient of the standardized inputs)
LMS_RATES = [0.003, 0.01, 0.03, 0.1, 0.3]  # (normalized: c / mean ||x||^2)
EH_RATES = [0.03, 0.1, 0.3, 1.0]
EH_NOISE = 0.5  # (the paper's)
NLMS_RATES = [0.03, 0.1, 0.3]  # (normalized per row: c / (||x||^2 + 1))
NEH_RATES = [0.01, 0.03, 0.1]
NEH_NOISES = [0.5, 1.5]
JOBS = int(os.environ.get("HAG_CANARY_RULES_JOBS", "1"))  # processes training the configurations
EPOCHS = [1, 3, 10, 30, 100, 300, 1000, 3000]  # (scores of the online rules after these numbers of epochs, the same run)
RLS_EPOCHS = [1]
MAX_EPOCHS = int(os.environ.get("HAG_CANARY_RULES_MAX_EPOCHS", EPOCHS[-1]))  # (of the online rules, at most: those of


# EPOCHS up to it)


def frame_songs_of(rules, fit):
    """Songs whose frames give the moments of song_data: fit if a rule reads the selection "pca_frames"."""
    return fit if any((rule_inputs(label)[2] or "").startswith("pca_frames") for label in rules) else ()


def configurations(rule, rate=None, max_epochs=None, noise=None, alpha=None, selection=None):
    """[(hyperparameters, epochs)] of a rule: a run of the online rules for each hyperparameters, scored after each
    number of epochs of EPOCHS (at most max_epochs) (LMS and EH: the learning rates of the grid, or the single one
    rate; EH: the exploration noise EH_NOISE, or noise; RLS: the grid of NORMALIZATIONS and RLS_ALPHAS, or the single
    alpha with centered inputs). With a selection of products (inputs already standardized): no normalization, the
    ridge coefficients RIDGES_STD."""
    if selection:
        if rule == "ridge":
            return [(dict(ridge=float(r)), [None]) for r in RIDGES_STD]
        unique = {}  # (the configurations which differ only by the normalization: once)
        for hyper, epochs in configurations(rule, rate, max_epochs, noise, alpha):
            unique.setdefault(tuple((k, v) for k, v in hyper.items() if k != "normalization"), epochs)
        return [(dict(hyper), epochs) for hyper, epochs in unique.items()]
    epochs = [e for e in EPOCHS if max_epochs is None or e <= max_epochs]
    if rule == "ridge":
        return [(dict(ridge=float(r)), [None]) for r in RIDGES]
    if rule == "rls":
        if alpha:
            return [(dict(normalization="center", alpha=float(alpha)), RLS_EPOCHS)]
        return [(dict(normalization=n, alpha=float(a)), RLS_EPOCHS) for n in NORMALIZATIONS for a in RLS_ALPHAS]
    if rule == "lms":
        return [(dict(normalization="standardize", rate=c), epochs) for c in ([rate] if rate else LMS_RATES)]
    if rule == "nlms":
        return [(dict(normalization="standardize", rate=c), epochs) for c in ([rate] if rate else NLMS_RATES)]
    if rule == "neh":
        return [(dict(normalization="standardize", rate=c, noise=v), epochs) for c in ([rate] if rate else NEH_RATES) for v in ([noise] if noise else NEH_NOISES)]
    return [(dict(normalization="standardize", rate=c, noise=noise or EH_NOISE), epochs)
            for c in ([rate] if rate else EH_RATES)]


def label_configurations(label, rate=None, max_epochs=None, noise=None, alpha=None):
    """configurations of the rule of a label of HAG_CANARY_READOUT_RULES."""
    rule, _, selection = rule_inputs(label)
    return configurations(rule, rate, max_epochs, noise, alpha, selection)


def accuracy_counts(scores, true, classes):
    with np.errstate(all="ignore"):
        return int((classes[np.nan_to_num(scores, nan=-np.inf).argmax(axis=1)] == true).sum())


def configuration_counts(tasks, all_rows, all_states, fit, val, true, classes, fold, moments=None):
    """{(label, configuration, epochs): correct predictions on the songs val} of the configurations tasks [(label,
    configuration index, hyperparameters, epochs)] trained on the songs fit (one process: the ridge configurations of
    a label together, their generalized readout fitted once)."""
    correct = {}
    for label, c, hyper, epochs in tasks:
        rule, order, selection = rule_inputs(label)
        rng = np.random.default_rng([CV_SEED + fold, c])
        if selection:
            rows, inputs = selected_rows(selection, all_rows, fit, moments, CV_SEED + fold)
            X_val = inputs(np.vstack([all_states[w] for w in val]))
        else:
            rows, states = normalized(hyper, all_rows, all_states, fit)
            X_val = np.vstack([states[w] for w in val])
        for epoch, node in trained(rule, hyper, rows, fit, rng, epochs, order, bool(selection)):
            correct[(label, c, epoch)] = accuracy_counts(node.run(X_val), true, classes)
    return correct


def choose(model, params, size, songs, train_songs, frame, classes, readout, rules, rate=None, max_epochs=None,
           noise=None, alpha=None):
    """({rule: (hyperparameters with "epochs", out-of-fold accuracy)} of the best configuration of each rule (labels of
    HAG_CANARY_READOUT_RULES), out-of-fold accuracy of each configuration and number of epochs), in the 3-fold
    cross-validation by song of the studies on the train songs (the configurations of configurations(rule, rate,
    max_epochs, noise, alpha); all the rules on the same reservoirs, drawn with the seed CV_SEED + k for fold k)."""
    correct, total = {}, 0
    for k, (fit_index, val_index) in enumerate(KFold(N_FOLDS, shuffle=True, random_state=FOLD_SEED).split(train_songs)):
        np.random.seed(CV_SEED + k)
        fit, val = [train_songs[i] for i in fit_index], [train_songs[i] for i in val_index]
        reservoirs = build(model, params, size, pretrain_phrases(songs, fit, frame))
        all_states, all_rows, moments = song_data(reservoirs, songs, fit + val, readout, classes,
                                                  frame_songs_of(rules, fit))
        true = np.concatenate([songs[w]["next"] for w in val])
        groups = []  # (the configurations of a ridge rule together, one group per configuration of the online rules)
        for label in rules:
            tasks = [(label, c, hyper, epochs) for c, (hyper, epochs)
                     in enumerate(label_configurations(label, rate, max_epochs, noise, alpha))]
            groups += [tasks] if rule_order(label)[0] == "ridge" else [[task] for task in tasks]
        for counts in Parallel(n_jobs=min(JOBS, len(groups)))(
                delayed(configuration_counts)(tasks, all_rows, all_states, fit, val, true, classes, k, moments)
                for tasks in groups):
            for key, count in counts.items():
                correct[key] = correct.get(key, 0) + count
        total += len(true)
    chosen = {}
    for rule in rules:
        (_, c, epoch), count = max(((key, count) for key, count in correct.items() if key[0] == rule),
                                   key=lambda item: item[1])
        configs = label_configurations(rule, rate, max_epochs, noise, alpha)
        chosen[rule] = (dict(configs[c][0], epochs=epoch), count / total)
    curves = pd.DataFrame([dict(rule=rule, **label_configurations(rule, rate, max_epochs, noise, alpha)[c][0],
                                epochs=epoch, cv_accuracy=count / total)
                           for (rule, c, epoch), count in correct.items()])
    return chosen, curves


# ---- rules study
RULES = ("lms", "nlms", "rls", "eh", "eh_np", "eh_np_local", "neh", "ridge:2", "nlms:2", "neh:2")
# range of the normalized learning rate of the online rules (readout_rules: c / mean ||x||^2, LMS stable below 2)
# (nlms, neh: normalized by row, c / (||x||^2 + 1), NLMS stable below 2; the same ranges on the monomials, "<rule>:<n>")
RULE_RATES = {"lms": (1e-3, 0.3), "eh": (1e-2, 1.0), "eh_np": (1e-2, 1.0), "eh_np_local": (1e-2, 1.0),
              "nlms": (1e-3, 1.0), "neh": (1e-3, 0.3)}
# range of the amplitude of the exploration noise of the EH rules (uniform in [-noise, noise] on each output, one-hot
# targets; Hoerzer et al. 2014: 0.5)
RULE_NOISES = {"eh": (0.05, 4.0), "eh_np": (0.05, 4.0), "eh_np_local": (0.05, 4.0), "neh": (0.05, 4.0)}
# range of the coefficient alpha of RLS (initial P = I / alpha: the ridge coefficient of the centered inputs). Below
# 1e-8 the recursion loses its precision (float64; bachue.models.readout_rules.rls). (Until 2026-11-19: 1e-14 to 1e-3,
# with the bias of reservoirpy's node, the lower bound reached: those studies are not comparable.)
RULE_ALPHAS = {"rls": (1e-8, 1.0)}
# number of epochs of the online rules chosen up to this one (HAG_CANARY_HPO_RULE_MAX_EPOCHS; 1000, the cap of the
# first studies, was always chosen by LMS and EH)
HPO_MAX_EPOCHS = int(os.environ.get("HAG_CANARY_HPO_RULE_MAX_EPOCHS", 3000))


# ---- the instances of the classification datasets (bachue.datasets.benchmarks)
def check_no_frames(rules):
    if frame_songs_of(rules, [0]):
        raise ValueError("pca_frames: the moments of all the frames are not computed here")


def choose_on_instances(model, params, size, data, rules, rate=None, max_epochs=MAX_EPOCHS, noise=None, alpha=None):
    """({rule: (hyperparameters with "epochs", accuracy of the cross-validation)}, accuracy of each configuration and
    number of epochs) on the train instances of a classification dataset data (bachue.datasets.benchmarks.load): the
    last state of each instance, a 3-fold cross-validation (reservoirs of fold k drawn with the seed CV_SEED + k, HAG
    learning on up to 500 instances of the fold), the configurations of label_configurations(rule, rate, max_epochs,
    noise, alpha)."""
    check_no_frames(rules)
    inputs, labels = list(data.train), np.asarray(data.Y_train).argmax(axis=1)
    classes = np.arange(np.asarray(data.Y_train).shape[1])
    correct = {}
    for k, (fit, val) in enumerate(KFold(N_FOLDS, shuffle=True, random_state=FOLD_SEED).split(inputs)):
        np.random.seed(CV_SEED + k)
        fit, val = [int(i) for i in fit], [int(i) for i in val]
        reservoirs = build(model, params, size, np.array([inputs[i] for i in fit[:500]], dtype=object))
        all_states, all_rows = instance_data(reservoirs, inputs, data.Y_train)
        groups = []  # (the configurations of a ridge rule together, one group per configuration of the online rules)
        for label in rules:
            tasks = [(label, c, hyper, epochs) for c, (hyper, epochs)
                     in enumerate(label_configurations(label, rate, max_epochs, noise, alpha))]
            groups += [tasks] if rule_order(label)[0] == "ridge" else [[task] for task in tasks]
        for counts in Parallel(n_jobs=min(JOBS, len(groups)))(
                delayed(configuration_counts)(tasks, all_rows, all_states, fit, val, labels[val], classes, k)
                for tasks in groups):
            for key, count in counts.items():
                correct[key] = correct.get(key, 0) + count
    chosen = {}
    for rule in rules:
        (_, c, epoch), count = max(((key, count) for key, count in correct.items() if key[0] == rule),
                                   key=lambda item: item[1])
        chosen[rule] = (dict(label_configurations(rule, rate, max_epochs, noise, alpha)[c][0], epochs=epoch),
                        count / len(inputs))
    curves = pd.DataFrame([dict(rule=rule, **label_configurations(rule, rate, max_epochs, noise, alpha)[c][0], epochs=epoch,
                                cv_accuracy=count / len(inputs)) for (rule, c, epoch), count in correct.items()])
    return chosen, curves


def objective_of_rule(rule):
    """objective_of of bachue.hpo.studies.run_study for the rule (a label of bachue.models.readout_rules.training), on
    the songs (the next phrase: the readout "last" -> "last") or on the instances of a classification dataset (the
    last state of each instance: choose_on_instances)."""
    base = rule_inputs(rule)[0]

    def objective_of(model, size, *data):
        def objective(trial):
            params = reservoir_parameters(trial, model, data[-1])
            rate = trial.suggest_float("readout:rate", *RULE_RATES[base], log=True) if base in RULE_RATES else None
            noise = trial.suggest_float("readout:noise", *RULE_NOISES[base], log=True) if base in RULE_NOISES else None
            alpha = trial.suggest_float("readout:alpha", *RULE_ALPHAS[base], log=True) if base in RULE_ALPHAS else None
            if len(data) == 2:  # (the instances of a classification dataset, bachue.hpo.studies.instances_data)
                chosen, _ = choose_on_instances(model, params, size, data[0], [rule], rate, HPO_MAX_EPOCHS, noise, alpha)
            else:  # (the songs, bachue.hpo.studies.next_phrase_data)
                songs, train_songs, frame, classes, _ = data
                chosen, _ = choose(model, params, size, songs, train_songs, frame, classes, BASE_READOUT, [rule], rate,
                                   HPO_MAX_EPOCHS, noise, alpha)
            hyper, accuracy = chosen[rule]
            trial.set_user_attr("readout_rule", hyper)
            return accuracy
        return objective
    return objective_of


if __name__ == "__main__":
    if HPO_RULE not in RULES:
        raise ValueError(f"HAG_CANARY_HPO_RULE={HPO_RULE}: one of {RULES} (on selected products: "
                         "bachue.hpo.partial_order2_study; the linear ridge readout: bachue.hpo.readout_window_study)")
    run_study(sys.argv[1], int(sys.argv[2]), objective_of_rule(HPO_RULE),
              lambda trial: f"readout {trial.user_attrs['readout_rule']}", tasks=("next_phrase", "classification"))
