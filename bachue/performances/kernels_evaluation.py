"""
Final evaluation of a study of bachue.hpo.kernel_study (BACHUE_KERNEL): the readout of its best trial (kernel
parameters and ridge coefficient) trained on all the train songs (instances) and scored on the test songs (instances).
The kernel draws nothing at random: a single run, no seeds. The test frames of norm above the largest norm M of the
train frames are brought back to the norm M (bachue.models.volterra_kernel.clipped).

Results: outputs/expansions/performances/<study>.csv, one row (the metrics of bachue.tasks.next_phrase.metrics on the
songs, the accuracy on a classification dataset; the parameters and the cross-validation accuracy of the trial).

Usage (the variables of the study):
  BACHUE_KERNEL=volterra BACHUE_DATASET=JapaneseVowels python -m bachue.performances.kernels_evaluation volterra 32
"""
import sys

import numpy as np
import pandas as pd

from bachue.common import OUTPUTS_DIR
from bachue.hpo.kernel_study import KERNELS, instance_rows, kernel_of, song_rows, train_rows
from bachue.hpo.studies import best_trial, instances_data, next_phrase_data, study_name_of
from bachue.models.kernel_ridge import KernelRidge
from bachue.models.volterra_kernel import clipped, max_norm, volterra_gram
from bachue.settings import DATASET_NAME, KERNEL, STUDY_KIND, TASK

PERFORMANCES = OUTPUTS_DIR / STUDY_KIND / "performances" / DATASET_NAME  # (one folder per dataset)


def run(model, window):
    name = study_name_of(model, window)
    trial = best_trial(name)
    if trial is None:
        raise ValueError(f"No study {name} with a completed trial")
    data = next_phrase_data() if TASK == "next_phrase" else instances_data()
    H, Y, _, classes, _ = train_rows(window, *data)
    norm = max_norm(H)
    if TASK == "next_phrase":
        songs, train_songs = data[0], set(data[1])
        test_songs = sorted(w for w in songs if w not in train_songs)
        H_test, _, true, _ = song_rows(songs, test_songs, classes, window)
    else:
        H_test, _, true = instance_rows(list(data[0].test), data[0].Y_test, window)
    tau, lam = kernel_of(trial.params, norm)
    ridge = trial.user_attrs["readout"]["ridge"]
    readout = KernelRidge(volterra_gram(H, H, tau, lam), Y)
    scores = readout.predict(volterra_gram(clipped(H_test, norm), H, tau, lam), ridge)
    if TASK == "next_phrase":
        from bachue.tasks.next_phrase import metrics
        result = metrics(scores, true, classes)
    else:
        result = dict(accuracy=float(np.mean(scores.argmax(axis=1) == true)))
    row = dict(dataset=DATASET_NAME, kernel=model, window=window, trial=trial.number, **trial.params, tau=tau,
               lam=lam, ridge=ridge, cv_accuracy=trial.value, n_train=len(H), n_test=len(H_test), **result)
    PERFORMANCES.mkdir(parents=True, exist_ok=True)
    path = PERFORMANCES / f"{name}.csv"
    pd.DataFrame([row]).to_csv(path, index=False)
    print(f"{name}: test accuracy {100 * result['accuracy']:.2f} % (cross-validation {100 * trial.value:.2f} %), {path}")
    return row


if __name__ == "__main__":
    model, window = sys.argv[1], int(sys.argv[2])
    if KERNEL not in KERNELS or model != KERNEL:
        raise ValueError(f"BACHUE_KERNEL={KERNEL} and the model {model}: the same kernel, one of {KERNELS}")
    run(model, window)
