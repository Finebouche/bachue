"""
Studies of a ridge readout on a kernel of the inputs, without reservoir (BACHUE_KERNEL): the reference without a
finite reservoir of the readouts of the other studies, searched as each of them (400 trials, the same folds and the
same objective). Kernel: "volterra", the Volterra reservoir kernel of Gonon, Grigoryeva & Ortega (2022)
(bachue.models.volterra_kernel: an infinite reservoir read by a readout on all the monomials of the inputs at all the
delays), on the MFCC frames of a window of W frames before each prediction point (the "size" of the study, in its
name "<prefix>volterra_<W>").

Rows, as the readout "last" -> "last" of the other studies: on the songs, one per prediction point (bachue.tasks.next_phrase,
with BACHUE_WITH_CALLS=1: the end of each phrase, calls included, and the point before the first phrase, a history of
zeros), labelled by the next phrase; on a classification dataset, the last frame of each instance. The frames of the
train songs (instances) set the norm M of the kernel (bachue.models.volterra_kernel.parameters).

Search: theta in THETAS (tau = theta / M: the weight tau^i of a monomial of degree i of a frame) and the memory 1 - mu
log-uniform in MEMORIES (lambda = mu sqrt(1 - theta^2): the weight lambda^j of the delay j); the ridge coefficient
(unpenalized bias) is chosen in KERNEL_RIDGES in the cross-validation (user attribute "readout" of the trial, with tau
and lambda). Objective: the out-of-fold accuracy of the 3-fold cross-validation (by song on the train songs, by instance
on the train instances, the folds of the other studies). The kernel of all the train rows is computed once per trial
(it has no fitted parameter), the readout of each fold solved in dual form (bachue.models.readout_rules.kernel_ridge).

Usage (final evaluation: bachue.performances.kernels_evaluation):
  BACHUE_KERNEL=volterra BACHUE_DATASET=JapaneseVowels python -m bachue.hpo.kernel_study volterra 32
  BACHUE_KERNEL=volterra BACHUE_TASK=next_phrase BACHUE_WITH_CALLS=1 python -m bachue.hpo.kernel_study volterra 600
"""
import sys

import numpy as np
from sklearn.model_selection import KFold

from bachue.hpo.studies import FOLD_SEED, run_study
from bachue.models.readout_rules.kernel_ridge import KernelRidge
from bachue.models.volterra_kernel import lagged_instances, lagged_points, max_norm, parameters, volterra_gram
from bachue.settings import CALLS, KERNEL, LEAK, N_FOLDS, TASK

KERNELS = ("volterra",)
KERNEL_RIDGES = 10.0 ** np.arange(-14, 7)  # (those of the other ridge readouts, RIDGES, and up to 1e6: the kernel can be large)
THETAS = (0.01, 0.99)
MEMORIES = (1e-4, 0.99)


def song_rows(songs, waves, classes, window):
    """(histories (rows, window, d), one-hot next labels, next labels, {wave: its rows}) of the prediction points of
    the songs (bachue.tasks.next_phrase.prepare: songs[w]["points"], songs[w]["next"])."""
    histories, Y, true, rows, start = [], [], [], {}, 0
    for w in waves:
        points = songs[w]["points"]
        histories.append(lagged_points(songs[w]["mfcc"], points, window))
        Y.append((songs[w]["next"][:, None] == classes[None, :]).astype(np.float64))
        true.append(songs[w]["next"])
        rows[w] = np.arange(start, start + len(points))
        start += len(points)
    return np.concatenate(histories), np.vstack(Y), np.concatenate(true), rows


def instance_rows(inputs, Y, window):
    """(histories (instances, window, d), one-hot labels, labels) of the last frame of each instance."""
    Y = np.asarray(Y, dtype=np.float64)
    return lagged_instances(inputs, window), Y, Y.argmax(axis=1)


def train_rows(window, *data):
    """(histories, one-hot labels, labels, classes, folds [(fit rows, val rows)]) of the train songs or instances of
    the data of the study (bachue.hpo.studies.next_phrase_data or instances_data), with the folds of the other
    studies (by song, or by instance)."""
    splitter = KFold(N_FOLDS, shuffle=True, random_state=FOLD_SEED)
    if TASK == "next_phrase":
        songs, train_songs, _, classes, _ = data
        H, Y, true, rows = song_rows(songs, train_songs, classes, window)
        folds = [(np.concatenate([rows[train_songs[i]] for i in fit]), np.concatenate([rows[train_songs[i]] for i in val]))
                 for fit, val in splitter.split(train_songs)]
        return H, Y, true, classes, folds
    benchmark = data[0]
    H, Y, true = instance_rows(list(benchmark.train), benchmark.Y_train, window)
    return H, Y, true, np.arange(Y.shape[1]), list(splitter.split(H))


def kernel_of(trial_params, norm):
    """(tau, lambda) of the parameters of a trial (theta, memory)."""
    return parameters(trial_params["theta"], 1 - trial_params["memory"], norm)


def cross_validation(K, Y, true, classes, folds, ridges=KERNEL_RIDGES):
    """(best ridge coefficient, its out-of-fold accuracy) of the readout on the kernel K of all the train rows."""
    correct = dict.fromkeys(ridges, 0)
    for fit, val in folds:
        readout = KernelRidge(K[np.ix_(fit, fit)], Y[fit])
        for ridge, outputs in readout.predictions(K[np.ix_(val, fit)], ridges):
            correct[ridge] += int((classes[outputs.argmax(axis=1)] == true[val]).sum())
    total = sum(len(val) for _, val in folds)
    ridge = max(ridges, key=lambda r: correct[r])  # (the smallest of the best: the first one)
    return float(ridge), correct[ridge] / total


def objective_of(model, window, *data):
    H, Y, true, classes, folds = train_rows(window, *data)
    norm = max_norm(H)

    def objective(trial):
        theta = trial.suggest_float("theta", *THETAS)
        memory = trial.suggest_float("memory", *MEMORIES, log=True)
        tau, lam = kernel_of(dict(theta=theta, memory=memory), norm)
        ridge, accuracy = cross_validation(volterra_gram(H, H, tau, lam), Y, true, classes, folds)
        trial.set_user_attr("readout", dict(ridge=ridge, tau=tau, lam=lam))
        return accuracy
    return objective


if __name__ == "__main__":
    model, window = sys.argv[1], int(sys.argv[2])
    if KERNEL not in KERNELS or model != KERNEL:
        raise ValueError(f"BACHUE_KERNEL={KERNEL} and the model {model}: the same kernel, one of {KERNELS}")
    if LEAK or (TASK == "next_phrase" and not CALLS):
        raise ValueError("The kernel studies: without leak (no reservoir), and on the songs with BACHUE_WITH_CALLS=1")
    run_study(model, window, objective_of, lambda trial: f"readout {trial.user_attrs['readout']}",
              tasks=("next_phrase", "classification"))
