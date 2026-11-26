"""
Final evaluation of the next-phrase prediction on the test songs (bachue.tasks.next_phrase: the task, its readouts
and its metrics), after the studies of bachue.hpo.readout_window_study: for each condition (conditions) and seed, the
reservoirs are drawn with the seed and learn (HAG) on phrases of the train songs, each readout is trained on all the
train songs with its ridge coefficient (that of the study, or chosen by the cross-validation of bachue.hpo.readout_window_study) and
scored on the test songs.

Conditions, for each SIZE (number of neurons in all; missing studies skipped):
  - "mfcc" (no reservoir, once);
  - the pure models (bachue.models.reservoirs.PURE_MODELS) with the best parameters of their study "<model>_<SIZE>"
    (named by the task, e.g. "next_phrase_leak_<model>_<SIZE>");
  - "<hag>_proj": the parameters of "<hag>_<SIZE>" without the HAG learning;
  - "<hag>+<esn>": hybrid of the best parts optimized alone with SIZE / 2 neurons (the hybrids are not optimized).
The paired comparisons of their results are made with the condition REFERENCE.

Results: outputs/<study>/performances/<prefix>hpo.csv (e.g. next_phrase_leak_hpo.csv; <prefix>hpo_order<n>.csv with the
generalized readout, <prefix>hpo_points_<points>[<tail>].csv with other training rows) and their paired comparisons
(bachue.analysis.comparisons): <...>_comparisons.csv, <...>_readout_comparisons.csv.
Usage (BACHUE_TASK=next_phrase, with the variables of the studies, e.g. BACHUE_LEAKY_RESERVOIRS=1):
  python -m bachue.performances.readout_window_evaluation [sizes] [nb_seeds]   all the conditions (default 500,1000, 10 seeds)
  python -m bachue.performances.readout_window_evaluation run <condition> <size> [nb_seeds]   one condition
                                             (slurm/submit_next_phrase.sh), results in outputs/<study>/performances/<name>_parts
  python -m bachue.performances.readout_window_evaluation merge                results of the parts and the comparisons
  python -m bachue.performances.readout_window_evaluation list <sizes>         "<condition> <size>" of all the conditions
  python -m bachue.performances.readout_window_evaluation baselines            the n-grams (bachue.performances.readout_window_evaluation)
"""
import sys
from collections import Counter

import numpy as np
import pandas as pd

from bachue.analysis.comparisons import compare, compare_readouts, filled_tails
from bachue.common import OUTPUTS_DIR, load_dataset, pretrain_phrases
from bachue.hpo.readout_window_study import choose_ridges
from bachue.hpo.studies import best_trial, study_name_of
from bachue.models.reservoirs import ESN_MODELS, HAG_MODELS, PURE_MODELS, build, parts
from bachue.models.readout_rules.ridge_readouts import READOUT_ORDER, fit_readout
from bachue.settings import CALLS, DATASET_NAME, HPO_READOUT, STUDY_PREFIX, TAIL_HPO, TASK
from bachue.tasks.next_phrase import (BASE_READOUT, READOUTS, START, decision_of, metrics, point_data, prepare,
                                      readout_name, readout_of_params, training_of)


# (one folder per dataset, e.g. outputs/readout_window/performances/marron1/)
OUTPUTS = OUTPUTS_DIR / "readout_window" / "performances" / DATASET_NAME
ANALYSIS = OUTPUTS_DIR / "readout_window" / "analysis" / DATASET_NAME  # (the paired comparisons)

SIZES = [500, 1000]
REFERENCE = "random_ee"  # (condition of the paired comparisons)


def conditions(size, study_name_of=study_name_of):
    """[(condition, model, params, ridge, threshold)] of size neurons (see the module docstring), the studies being
    named by study_name_of(model, size) (default those of the task, bachue.hpo.studies.study_name_of): ridge and
    threshold None when chosen by cross-validation, threshold None without threshold."""
    result = []

    def from_study(condition, model, study_name):
        trial = best_trial(study_name)
        if trial is None:
            print(f"  (no study {study_name}: {condition} skipped)")
        else:
            result.append((condition, model, trial.params, trial.user_attrs.get("ridge"), trial.user_attrs.get("threshold")))

    for model in PURE_MODELS:
        from_study(model, model, study_name_of(model, size))
    for hag in HAG_MODELS:
        trial = best_trial(study_name_of(hag, size))
        if trial is not None:
            result.append((f"{hag}_proj", f"{hag}_proj", trial.params, None, None))
    for hag in HAG_MODELS:
        for esn in ESN_MODELS:
            hybrid = f"{hag}+{esn}"
            halves = [best_trial(study_name_of(part, size // 2)) for part, _ in parts(hybrid)]
            if all(trial is not None for trial in halves):
                params = {prefix + key: value for (_, prefix), trial in zip(parts(hybrid), halves)
                          for key, value in trial.params.items()}
                result.append((hybrid, hybrid, params, None, None))
    return result


def evaluate(model, params, size, ridges, songs, train_songs, test_songs, frame, classes, seed):
    """{readout: metrics on the test songs}, the readouts (with their coefficients ridges) on the same reservoirs, each
    readout trained once for all its decisions."""
    np.random.seed(seed)
    reservoirs = build(model, params, size, pretrain_phrases(songs, train_songs, frame))
    states, data = point_data(reservoirs, songs, train_songs + test_songs, list(ridges), train_songs, classes)
    true = np.concatenate([songs[w]["next"] for w in test_songs])
    fitted = {}
    for readout, ridge in ridges.items():
        if (training_of(readout), ridge) not in fitted:
            fitted[training_of(readout), ridge] = fit_readout(data[training_of(readout)], ridge)
    return {readout: metrics(fitted[training_of(readout), ridge].run(
        np.vstack([states[decision_of(readout)][w] for w in test_songs])), true, classes)
            for readout, ridge in ridges.items()}


NAME = (f"{STUDY_PREFIX}hpo" + (f"_order{READOUT_ORDER}" if READOUT_ORDER > 1 else "")
        + ("" if HPO_READOUT else readout_name(READOUTS[0]) if len(READOUTS) == 1 else "_readouts"))
PARTS = OUTPUTS / f"{NAME}_parts"


def all_conditions(sizes):
    """[(condition, model, params, ridge, size)], mfcc first (size 0); ridge None when chosen by cross-validation."""
    todo = [("mfcc", "mfcc", {}, None, 0)]
    for size in sizes:
        todo += [(condition, model, params, ridge, size) for condition, model, params, ridge, _ in conditions(size)]
    return todo


def run(todo, nb_seeds):
    """Results of the readouts (READOUTS, columns points, decision and tail when they are not "last" -> "last" alone)
    for each condition and seed."""
    songs, train_songs, test_songs, frame = load_dataset(top_db=None)
    classes = prepare(songs, train_songs, frame)
    print(f"{len(train_songs)} train / {len(test_songs)} test songs, {len(classes)} classes, "
          f"{sum(len(songs[w]['next']) for w in test_songs)} test points", flush=True)
    rows = []
    for condition, model, params, ridge, size in todo:
        # (the coefficient of the study for its readout, "last" -> "last" or HPO_READOUT: that of the linear readout)
        study_readout = READOUTS[0] if HPO_READOUT else BASE_READOUT
        study = ridge is not None and READOUT_ORDER == 1 and study_readout in READOUTS and not TAIL_HPO
        chosen = {} if study and READOUTS == [study_readout] else \
            choose_ridges(model, params, size, songs, train_songs, frame, classes, READOUTS)
        chosen.update({study_readout: (ridge, np.nan)} if study else {})
        readouts, columns_of = READOUTS, {}
        if TAIL_HPO:  # "last" -> "last", the readout of the study and the best tail of each kind of readout
            hpo = readout_of_params(params) or max(READOUTS, key=lambda readout: chosen[readout][1])
            best_tail = {max((r for r in READOUTS[1:] if r[:2] == kind), key=lambda r: chosen[r][1])
                         for kind in dict.fromkeys(r[:2] for r in READOUTS[1:])}
            readouts = list(dict.fromkeys([BASE_READOUT, hpo, *sorted(best_tail)]))
            columns_of = {r: dict(hpo=r == hpo, best_tail=r in best_tail) for r in readouts}
        ridges = {readout: chosen[readout][0] for readout in readouts}
        accuracies = {readout: [] for readout in readouts}
        for seed in range(nb_seeds):
            results = evaluate(model, params, size, ridges, songs, train_songs, test_songs, frame, classes, seed)
            for readout, result in results.items():
                points, decision, tail, decision_tail = readout
                columns = {} if READOUTS == [BASE_READOUT] else dict(points=points, decision=decision, tail=tail,
                                                                     decision_tail=decision_tail,
                                                                     **columns_of.get(readout, {}))
                rows.append(dict(condition=condition, network_size=size, seed=seed, **columns,
                                 ridge=ridges[readout], cv_accuracy=chosen[readout][1], **result))
                accuracies[readout].append(result["accuracy"])
        for readout, accuracy in accuracies.items():
            tails = [t for t in dict.fromkeys(readout[2:]) if t is not None]
            print(f"{condition} ({size}) {readout[0]} -> {readout[1]}{f' ({'/'.join(map(str, tails))})' if tails else ''}:"
                  f" accuracy {100 * np.mean(accuracy):.2f} ± {100 * np.std(accuracy):.2f} % "
                  f"(ridge {ridges[readout]:.0e})", flush=True)
    return pd.DataFrame(rows)


def save(results):
    """The results, the paired comparisons of the conditions (for each readout) and, with several readouts, those of
    the readouts with "last" -> "last"."""
    pd.set_option("display.width", 200)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUTS / f"{NAME}.csv", index=False)
    if "points" not in results:
        comparisons = compare(results, "accuracy", REFERENCE)
    elif "hpo" in results:  # (the readouts of the studies, TAIL_HPO; the tail differs between the conditions)
        comparisons = compare(results[results.hpo.astype(bool)], "accuracy", REFERENCE)
        readout_comparisons = compare_readouts(results[results.hpo.astype(bool) | ((results.points == "last")
                                                                                   & (results.decision == "last"))])
        print(readout_comparisons.round(4).to_string(index=False))
        ANALYSIS.mkdir(parents=True, exist_ok=True)
        readout_comparisons.to_csv(ANALYSIS / f"{NAME}_readout_comparisons.csv", index=False)
        print("Saved to", ANALYSIS / f"{NAME}_readout_comparisons.csv")
    else:
        comparisons = pd.concat([compare(group, "accuracy", REFERENCE).assign(
            points=points, decision=decision, tail=None if tail < 0 else int(tail),
            decision_tail=None if decision_tail < 0 else int(decision_tail))
            for (points, decision, tail, decision_tail), group in filled_tails(results).groupby(
                ["points", "decision", "tail", "decision_tail"])], ignore_index=True)
        readout_comparisons = compare_readouts(results)
        print(readout_comparisons.round(4).to_string(index=False))
        ANALYSIS.mkdir(parents=True, exist_ok=True)
        readout_comparisons.to_csv(ANALYSIS / f"{NAME}_readout_comparisons.csv", index=False)
        print("Saved to", ANALYSIS / f"{NAME}_readout_comparisons.csv")
    print(comparisons.round(4).to_string(index=False))
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    comparisons.to_csv(ANALYSIS / f"{NAME}_comparisons.csv", index=False)
    print("Saved to", OUTPUTS / f"{NAME}.csv", "and", ANALYSIS / f"{NAME}_comparisons.csv")


# ---- References of the next-phrase prediction: n-grams of the true labels of the previous phrases, of fixed order n (the n
# previous labels, START before the first phrase; unseen context: the most frequent next label) and adaptive (the
# longest context seen in the train songs, up to MAX_ORDER), scored on the test songs.
# Results: outputs/<study>/performances/next_phrase_ngrams.csv (next_phrase_calls_ngrams.csv with the calls).
# Usage: python -m bachue.performances.readout_window_evaluation baselines
MAX_ORDER = 10


# ------------------------------------------------- n-grams --------------------------------------------------------
def baselines():
    """For each order (None: adaptive), metrics of the scores (frequencies of the next labels) of the n-gram at the
    test points, and the share of the points whose context was not seen in the train songs (marginal frequencies
    then)."""
    songs, train_songs, test_songs, frame = load_dataset(top_db=None)
    classes = prepare(songs, train_songs, frame)
    true = np.concatenate([songs[w]["next"] for w in test_songs])
    index = {label: i for i, label in enumerate(classes)}

    def contexts(song, n):
        history = [START] * n + [label for _, _, label in song["next_phrases"]]
        return [tuple(history[j:j + n]) for j in range(len(song["next"]))]

    rows = []
    for order in [*range(0, MAX_ORDER + 1), None]:
        orders = list(range(MAX_ORDER, -1, -1)) if order is None else sorted({order, 0}, reverse=True)
        counts = {n: {} for n in orders}
        for w in train_songs:
            for n in orders:
                for context, label in zip(contexts(songs[w], n), songs[w]["next"]):
                    counts[n].setdefault(context, Counter())[label] += 1
        scores, unseen = [], 0
        for w in test_songs:
            by_order = {n: contexts(songs[w], n) for n in orders}
            for j in range(len(songs[w]["next"])):
                n = next(n for n in orders if by_order[n][j] in counts[n])
                unseen += n == 0 and (order is None or order > 0)
                row = np.zeros(len(classes))
                for label, count in counts[n][by_order[n][j]].items():
                    row[index[label]] = count
                scores.append(row / row.sum())
        rows.append(dict(condition="ngram_adaptive" if order is None else f"ngram_{order}", order=order,
                         unseen_context=unseen / len(scores), **metrics(np.array(scores), true, classes)))
    results = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(results.round(4).to_string(index=False))
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    file_name = OUTPUTS / f"next_phrase_{'calls_' * CALLS}ngrams.csv"
    results.to_csv(file_name, index=False)
    print("Saved to", file_name)


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "baselines":
        baselines()
    elif TASK != "next_phrase":
        raise RuntimeError("BACHUE_TASK=next_phrase is needed (names of the studies)")
    elif command == "list":
        for condition, _, _, _, size in all_conditions([int(s) for s in sys.argv[2].split(",")]):
            print(condition, size)
    elif command == "run":
        condition, size = sys.argv[2], int(sys.argv[3])
        nb_seeds = int(sys.argv[4]) if len(sys.argv) > 4 else 10
        todo = [c for c in all_conditions([size] if size else []) if c[0] == condition and c[4] == size]
        if not todo:
            raise ValueError(f"Unknown condition {condition} ({size})")
        PARTS.mkdir(parents=True, exist_ok=True)
        run(todo, nb_seeds).to_csv(PARTS / f"{condition}_{size}.csv", index=False)
    elif command == "merge":
        part_files = sorted(PARTS.glob("*.csv"))
        print(f"{len(part_files)} conditions")
        save(pd.concat([pd.read_csv(part) for part in part_files], ignore_index=True))
    else:
        sizes = [int(s) for s in sys.argv[1].split(",")] if len(sys.argv) > 1 else SIZES
        nb_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        save(run(all_conditions(sizes), nb_seeds))
