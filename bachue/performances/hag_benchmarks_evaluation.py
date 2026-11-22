"""
Learning rules of the readout on the classification datasets of hag (bachue.datasets.benchmarks: JapaneseVowels, FSDD,
...), with the rules, the inputs and the selections of bachue.performances.learning_rules_evaluation: "ridge", "rls",
"nlms", "eh_np_local", ... on the states, on all their monomials ("<rule>:2") or on the states and selected products
("<rule>:<selection>", e.g. "nlms:pca:32", "ridge:random_pairs:n").

The readout reads the last state of the reservoir on each instance, run from a zero state (the "sequence-to-vector"
readout of HAG's evaluations, the "last" -> "last" of the canary songs), and outputs a score for each class.

Reservoirs: those of HAG's studies of the dataset (hag.hpo.utility.retrieve_best_model: 400 trials, optimized for
HAG's own ridge readout on the last states), built by bachue.models.reservoirs.build. Exploratory, as
bachue.performances.learning_rules_evaluation without BACHUE_STUDY_RULE: all the rules read the same reservoirs,
optimized for the ridge readout and not for each rule: no fair ranking of the rules.

Protocol of readout_rules: the hyperparameters of each rule (its grid: readout_rules.configurations; number of epochs
at most BACHUE_ONLINE_MAX_EPOCHS) chosen by a 3-fold cross-validation on the train instances (reservoirs of fold k
drawn with the seed CV_SEED + k, HAG learning on the instances of the fold); then each rule trained on all the train
instances and scored on the test instances for each seed (reservoirs drawn with the seed).

Results: outputs/benchmarks/<dataset>_<model>_readout_rules.csv (one row per seed and rule) and ..._cv.csv (accuracy
of each configuration and number of epochs in the cross-validation).
Usage: python -m bachue.performances.hag_benchmarks_evaluation <dataset> [model] [nb_seeds]     (model: default random_ee; rules:
  BACHUE_BENCHMARK_RULES, comma-separated)
"""
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import f1_score

from bachue.common import OUTPUTS_DIR
from bachue.datasets.benchmarks import instance_data, load
from bachue.hpo.learning_rules_study import JOBS, MAX_EPOCHS, choose_on_instances
from bachue.models.readout_rules.training import normalized, rule_inputs, rule_order, selected_rows, trained
from bachue.models.reservoirs import build
from bachue.settings import env


# ---- Reservoirs and hyperparameters of the rules on the classification datasets of hag: the parameters of the best
# trial of HAG's study of a dataset (study_parameters), and the hyperparameters of each rule chosen by a 3-fold
# cross-validation on the train instances (choose). The first step of bachue.performances.hag_benchmarks_evaluation (which
# describes the protocol).
RULES = env("BACHUE_BENCHMARK_RULES", "ridge,rls,nlms,eh_np_local,ridge:2,nlms:2,ridge:random_pairs:n,"
                       "nlms:random_pairs:n,ridge:pca:32,nlms:pca:32").split(",")


def study_parameters(dataset, model, is_multivariate=True):
    """(hyperparameters, number of units) of the best trial of HAG's study of the model on the dataset."""
    from hag.hpo.utility import retrieve_best_model, study_db_dir
    study = retrieve_best_model(model, dataset, is_multivariate, variate_type="multi", data_type="normal", prefix="tpe",
                                db_dir=study_db_dir(dataset), verbosity=0)
    params = dict(study.best_params)
    return params, int(params["network_size"])


def choose(model, params, size, data, rules=RULES, max_epochs=MAX_EPOCHS):
    """bachue.hpo.learning_rules_study.choose_on_instances with the grids of the rules (see the module docstring)."""
    return choose_on_instances(model, params, size, data, rules, max_epochs=max_epochs)


# ---- benchmark rules
RESULTS = OUTPUTS_DIR / "hag_benchmarks" / "performances"


def seed_scores(label, hyper, train_rows, train_states, test_states, true, seed):
    """dict(accuracy, f1_macro) on the test instances of the rule label with its chosen hyperparameters hyper, trained
    on the train instances."""
    rule, order, selection = rule_inputs(label)
    config = {key: value for key, value in hyper.items() if key != "epochs"}
    fit, test = sorted(train_rows), sorted(test_states)
    if selection:
        rows, inputs = selected_rows(selection, train_rows, fit, None, seed)
        X_test = inputs(np.vstack([test_states[i] for i in test]))
    else:
        rows, normalized_states = normalized(config, train_rows, {**train_states, **{("test", i): x for i, x in
                                                                                     test_states.items()}}, fit)
        X_test = np.vstack([normalized_states[("test", i)] for i in test])
    epochs = [hyper["epochs"]] if hyper["epochs"] is not None else [None]
    _, node = next(trained(rule, config, rows, fit, np.random.default_rng([seed, 1]), epochs, order, bool(selection)))
    with np.errstate(all="ignore"):
        predicted = np.nan_to_num(np.asarray(node.run(X_test)), nan=-np.inf).argmax(axis=1)
    return dict(accuracy=float(np.mean(predicted == true)),
                f1_macro=float(f1_score(true, predicted, average="macro", zero_division=0)))


def run(dataset, model="random_ee", nb_seeds=10, rules=RULES, max_epochs=MAX_EPOCHS):
    """(results of the rules for each seed, their cross-validation) on a classification dataset of hag."""
    data = load(dataset)
    if not data.is_instances_classification:
        raise ValueError(f"{dataset}: a {data.kind} dataset, only the classification ones are handled")
    params, size = study_parameters(dataset, model, data.is_multivariate)
    chosen, curves = choose(model, params, size, data, rules, max_epochs)
    for rule, (hyper, accuracy) in chosen.items():
        print(f"  {rule}: {hyper}, cross-validation accuracy {100 * accuracy:.2f} %", flush=True)
    true = np.asarray(data.Y_test).argmax(axis=1)
    seed_data = []
    for seed in range(nb_seeds):
        np.random.seed(seed)
        reservoirs = build(model, params, size, data.pretrain)
        train_states, train_rows = instance_data(reservoirs, data.train, data.Y_train)
        test_states, _ = instance_data(reservoirs, data.test, data.Y_test)
        seed_data.append((train_rows, train_states, test_states))
    tasks = [(seed, rule) for seed in range(nb_seeds) for rule in chosen]
    scores = Parallel(n_jobs=min(JOBS, len(tasks)))(
        delayed(seed_scores)(rule, chosen[rule][0], *seed_data[seed], true, seed) for seed, rule in tasks)
    keys = dict(dataset=dataset, condition=model, network_size=size)
    rows = [dict(**keys, seed=seed, rule=rule, order=rule_order(rule)[1], selection=rule_inputs(rule)[2] or "",
                 **chosen[rule][0], cv_accuracy=chosen[rule][1], **score) for (seed, rule), score in zip(tasks, scores)]
    return pd.DataFrame(rows), curves.assign(**keys)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m bachue.performances.hag_benchmarks_evaluation <dataset> [model] [nb_seeds]")
    dataset, model = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "random_ee"
    results, curves = run(dataset, model, int(sys.argv[3]) if len(sys.argv) > 3 else 10)
    RESULTS.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS / f"{dataset}_{model}_readout_rules.csv", index=False)
    curves.to_csv(RESULTS / f"{dataset}_{model}_readout_rules_cv.csv", index=False)
    summary = results.groupby("rule").accuracy.agg(["mean", "std"]).mul(100).round(2)
    print(summary.loc[[rule for rule in RULES if rule in summary.index]].to_string())
    print("Saved to", RESULTS / f"{dataset}_{model}_readout_rules.csv")
