"""
Study 1: the readout of the reservoirs, how its training rows and its decision read the states of the window of each
prediction point (bachue.tasks.next_phrase.readout_of). One study per readout, so that the readouts are compared on
reservoirs optimized for each of them (results: bachue.performances.readout_window_evaluation with the same variables).

Readout of the study: HAG_CANARY_HPO_READOUT=<points>:<decision>[:<tail>] (studies
"next_phrase_calls_[leak_]readout-<points>-<decision>[-<tail>]_<model>_<size>"; e.g. "tail_mean:tail_mean:10",
"mean:mean", "all:last", "tail:tail_mean:5"), or, without it, "last" -> "last" (the last state of each point for the
training and the decision: the studies "next_phrase_calls_[leak_]<model>_<size>", the reservoirs of the linear ridge
readout, the reference of the other studies).
  - training rows: "last" (the last state), "mean" (the mean of the states of the window), "all" (all its states),
    "tail" (its last <tail> states), "tail_mean" (their mean);
  - decision: "last", "mean" or "tail_mean" (the outputs on the mean of the last <tail> states).

Search: the parameters of the reservoirs (bachue.hpo.studies.reservoir_parameters, 400 trials by default). Objective:
the out-of-fold accuracy of the next phrase in the 3-fold cross-validation by song on the train songs, the ridge
coefficient of the readout chosen in this cross-validation (bachue.hpo.readout_window_study.choose_ridges; user attribute "ridge" of
the trial), not in the search.

Usage (HAG_CANARY_TASK=next_phrase HAG_CANARY_CALLS=1 [HAG_CANARY_LEAK=1] [HAG_CANARY_HPO_READOUT=...]):
  python -m bachue.hpo.readout_window_study <model> <size>     (slurm/submit_hpo_readouts.sh: all the readouts)
"""
import sys

import numpy as np
from sklearn.model_selection import KFold

from bachue.common import RIDGES, pretrain_phrases
from bachue.hpo.studies import CV_SEED, FOLD_SEED, reservoir_parameters, run_study
from bachue.models.reservoirs import build
from bachue.models.ridge_readouts import fitted_readouts
from bachue.settings import HPO_RULE, N_FOLDS
from bachue.tasks.next_phrase import READOUTS, decision_of, point_data, training_of


# ---- Ridge coefficient of the readouts of the next-phrase prediction, chosen by the 3-fold cross-validation by song
# on the train songs (out-of-fold accuracy; reservoirs of fold k drawn with the seed CV_SEED + k): the objective of
# the studies of bachue.hpo.readout_window_study, and the coefficients of the conditions without study in
# bachue.performances.readout_window_evaluation.
def choose_ridges(model, params, size, songs, train_songs, frame, classes, readouts):
    """{readout: (ridge coefficient of the best out-of-fold accuracy, this accuracy)}, in a 3-fold cross-validation by
    song (the folds of the studies), the readouts on the same reservoirs."""
    correct = {readout: np.zeros(len(RIDGES)) for readout in readouts}
    total = 0
    for k, (fit_index, val_index) in enumerate(KFold(N_FOLDS, shuffle=True, random_state=FOLD_SEED).split(train_songs)):
        np.random.seed(CV_SEED + k)
        fit, val = [train_songs[i] for i in fit_index], [train_songs[i] for i in val_index]
        reservoirs = build(model, params, size, pretrain_phrases(songs, fit, frame))
        states, data = point_data(reservoirs, songs, fit + val, readouts, fit, classes)
        true = np.concatenate([songs[w]["next"] for w in val])
        X_val = {decision: np.vstack([result[w] for w in val]) for decision, result in states.items()}
        for training, training_data in data.items():  # (each readout trained once, for all its decisions)
            for r, (_, fitted) in enumerate(fitted_readouts(training_data, RIDGES)):
                for readout in readouts:
                    if training_of(readout) == training:
                        scores = fitted.run(X_val[decision_of(readout)])
                        correct[readout][r] += (classes[scores.argmax(axis=1)] == true).sum()
        total += len(true)
    return {readout: (float(RIDGES[int(np.argmax(c))]), c.max() / total) for readout, c in correct.items()}


# ---- readouts study
def objective_of(model, size, songs, train_songs, frame, classes, length):  # (the data of next_phrase_data)
    readout = READOUTS[0]  # (that of HAG_CANARY_HPO_READOUT, or "last" -> "last")

    def objective(trial):
        params = reservoir_parameters(trial, model, length)
        ridge, accuracy = choose_ridges(model, params, size, songs, train_songs, frame, classes, [readout])[readout]
        trial.set_user_attr("ridge", ridge)
        return accuracy
    return objective


if __name__ == "__main__":
    if HPO_RULE:
        raise ValueError("HAG_CANARY_HPO_RULE: a study of a rule, bachue.hpo.learning_rules_study or bachue.hpo.partial_order2_study")
    run_study(sys.argv[1], int(sys.argv[2]), objective_of, lambda trial: f"ridge {trial.user_attrs['ridge']:.0e}")
