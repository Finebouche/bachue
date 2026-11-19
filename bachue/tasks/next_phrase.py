"""
Prediction of the next phrase of canary songs from their audio, in the configuration of Maxime Canonne's next-phrase
prediction (ESN reading the phrase labels of the song, with START and END), but with the MFCC of the song as input:
the reservoir runs over the whole song (no reset between the phrases) and, at the end of each phrase, the readout predicts
the label of the next one (END after the last phrase); before the first phrase (Maxime's START), it predicts the first
one. Phrases as in bachue.datasets.load_categorical_forecasting.load_songs: SIL, TRASH and call ignored, repeated labels merged; with
HAG_CANARY_CALLS=1, the calls are phrases too (SIL and TRASH ignored). With HAG_CANARY_RESET=1, the reservoirs hear
the current phrase only, from a zero state (as the Canary task of hag), instead of the whole song: the
state of a point is the last one of a run on the frames of the phrase that ends there (the zero state before the first
phrase); the HAG learning is the same (on phrases). Input: the 13 MFCC, without their deltas (their effect, with and
without leak: outputs/<study>/performances/ablation_deltas_leak).

No input after the prediction point: the MFCC are computed without the floor of the dB at 80 dB below the maximum of
the whole song (librosa's default, which depends on the future of each frame: bachue.common.load_dataset(top_db=None)),
and the MFCC frames are centered (half a window, 11.6 ms, of audio after their time), so the state used to predict
what follows the time t is the one of the last frame whose window ends before t (DELAY earlier).

Readout: ridge regression on the one-hot next label at the prediction points (reservoirpy.nodes.Ridge, solver
HAG_CANARY_RIDGE_SOLVER, default "svd": the singular value decomposition of the centered states, stable with the
smallest coefficients), its coefficient chosen by the 3-fold cross-validation by song on the train songs (out-of-fold
accuracy), then trained on all the train songs and scored on the test songs for each seed. With HAG_CANARY_READOUT_ORDER=n (n >= 2), the generalized
readout of Ohkubo & Inubushi (2024) instead (bachue.models.generalized_readout.GeneralizedRidge: ridge regression on all the
monomials of the state up to degree n, the quadratic form for n = 2), its coefficient always chosen by the same
cross-validation (the coefficients of the studies are those of the linear readout).
Training rows of the readout (HAG_CANARY_POINTS), from the states of the window of each point: those of the frames of
the phrase ending there, up to the point (from the first frame of the song for the point before the first phrase; the
state at the point alone when there is no such frame): "last" (default) its last state only, "tail" those of its
last HAG_CANARY_TAIL frames (default 10, 116 ms), "tail_mean" their mean, "all" all its states, "mean" their mean; all
the rows of a point have its next label.
Decision (HAG_CANARY_DECISION, the scores of the validation and test points): the outputs on the last state of the
window ("last"), on the mean of the states of its last HAG_CANARY_TAIL frames ("tail_mean", the mean of the outputs
on these states with the linear readout) or on the mean of all its states ("mean"); by default, on the training row of
the point, or on the mean of its rows ("tail": "tail_mean", "all": "mean").
With the studies of a fixed readout (HAG_CANARY_HPO_READOUT, bachue.hpo.readout_window_study: reservoirs optimized
for it), this readout alone, with the ridge coefficient of the study.
Comma-separated lists of these three variables: all their combinations (readouts (points, decision, tail), the tail
None when neither uses it), on the same reservoirs, their coefficients chosen by the same cross-validation (that of
the study for "last" -> "last"), and their paired comparisons with "last" -> "last".
With the studies of the readout (HAG_CANARY_TAIL_HPO=1, the studies of canopy), the readouts of each
condition are instead: "last" -> "last", that of its study (column hpo; for the conditions without study, mfcc and
hybrids, the best of TAIL_READOUTS in the cross-validation) and, for each kind of tail readout (training rows "tail"
or "tail_mean", decision "last" or "tail_mean"), the length of tail of the best cross-validation accuracy (column
best_tail), the coefficients of all of them chosen by the same cross-validation. With HAG_CANARY_TAIL_HPO=2 (TAIL_SEPARATE),
the tails of the training rows (tail) and of the decision (decision_tail) are two parameters of the studies, and
best_tail is the best pair of tails of each kind of readout; HAG_CANARY_DECISION_TAIL does the same for lists of
readouts (default: the tail of the training rows).
Metrics as Maxime's: accuracy and macro F1 of all the points,
without the last one of each song ("skip_last": its target is END), and of the phrases only ("no_end": points whose
target is a phrase, argmax over the phrases).
References: n-grams of the true labels of the previous phrases (bachue.tasks.next_phrase baselines): fixed order n
(the n previous labels, START before the first phrase; unseen context: the most frequent next label) and adaptive (the
longest context seen in the train songs, up to MAX_ORDER).

Results: outputs/<study>/performances/<HAG_CANARY_TASK=next_phrase prefix>hpo.csv (e.g. next_phrase_leak_hpo.csv) and
<...>_comparisons.csv (<...>hpo_order<n>.csv with the generalized readout, <...>hpo_points_<points>[<tail>].csv
with other training rows),
references in next_phrase_ngrams.csv (next_phrase_calls_ngrams.csv with the calls).
Usage (HAG_CANARY_TASK=next_phrase, with the variables of the studies, e.g. HAG_CANARY_LEAK=1):
  python -m bachue.performances.readout_window_evaluation [sizes] [nb_seeds]   all the conditions (default 500,1000, 10 seeds)
  python -m bachue.performances.readout_window_evaluation run <condition> <size> [nb_seeds]   one condition (slurm/submit_next_phrase.sh),
                                                                          results in outputs/<study>/performances/<name>_parts
  python -m bachue.performances.readout_window_evaluation merge                               results of the parts and the comparisons
  python -m bachue.performances.readout_window_evaluation list <sizes>                        "<condition> <size>" of all the conditions
  python -m bachue.performances.readout_window_evaluation baselines                           the n-grams
"""
import os

import numpy as np
from sklearn.metrics import f1_score

from bachue.datasets.load_categorical_forecasting import SONGS_SAMPLING_RATE, annotations_of_songs, canary_mfcc_config, song_phrases
from bachue.models.reservoirs import CHUNK, state_chunks, states
from bachue.models.ridge_readouts import READOUT_ORDER
from bachue.settings import CALLS, HPO_READOUT, MAX_TAIL, RESET, TAIL_HPO, TAIL_SEPARATE

START, END = "START", "END"
N_STATIC = 13  # MFCC, before their deltas and delta-deltas in the cached features
TAIL_MODES = ("tail", "tail_mean")  # (those of the last HAG_CANARY_TAIL frames of the windows)
WHOLE_MODES = ("mean", "all")  # (those of all the frames of the windows)
BASE_READOUT = ("last", "last", None, None)  # (training rows, decision, tail, decision tail): the last state only


def readout_of(points, decision, tail, decision_tail=None):
    """The readout (points, decision, tail, decision tail), the decision by default that of the training rows, the tail
    None when neither uses it, the tail of the decision by default that of the training rows (None for the decision on
    the last state)."""
    decision = decision or {"tail": "tail_mean", "all": "mean"}.get(points, points)
    if points not in ("last",) + WHOLE_MODES + TAIL_MODES or decision not in ("last", "mean", "tail_mean"):
        raise ValueError(f"Unknown readout {points} -> {decision}: HAG_CANARY_POINTS last, mean, all, tail or "
                         f"tail_mean, HAG_CANARY_DECISION last, mean or tail_mean")
    return (points, decision, tail if points in TAIL_MODES or decision == "tail_mean" else None,
            (decision_tail or tail) if decision == "tail_mean" else None)


def readout_of_params(params):
    """Readout of the parameters "readout:*" of a study of the readout (canopy's studies,
    TAIL_HPO), None without them: "last" -> "last" for a tail of 1 frame. With "readout:decision_tail" (TAIL_SEPARATE),
    the tail of the training rows ("readout:tail", 1: the last state) and that of the decision are separate."""
    tail = params.get("readout:tail")
    if tail is None:
        return None
    decision_tail = params.get("readout:decision_tail")
    if decision_tail is not None:
        points = "last" if tail == 1 else "tail_mean" if params["readout:train_mean"] else "tail"
        return readout_of(points, "last" if decision_tail == 1 else "tail_mean", int(tail) if tail > 1 else None,
                          int(decision_tail) if decision_tail > 1 else None)
    if tail == 1:
        return BASE_READOUT
    return readout_of("tail_mean" if params["readout:train_mean"] else "tail",
                      "tail_mean" if params["readout:decision_mean"] else "last", int(tail))


# all the readouts of the studies of the readout (TAIL_HPO), "last" -> "last" first
# (TAIL_SEPARATE: all the pairs of a training tail, "last" or 2 to MAX_TAIL frames, and a decision tail)
TAIL_READOUTS = list(dict.fromkeys(
    [BASE_READOUT] + ([readout_of(points, decision, tail, decision_tail)
                       for points, tail in [("last", None)] + [(m, t) for m in TAIL_MODES for t in range(2, MAX_TAIL + 1)]
                       for decision, decision_tail in [("last", None)] + [("tail_mean", t) for t in range(2, MAX_TAIL + 1)]]
                      if TAIL_SEPARATE else [readout_of(points, decision, tail) for points in TAIL_MODES
                                             for decision in ("last", "tail_mean") for tail in range(2, MAX_TAIL + 1)])))
# (HAG_CANARY_DECISION_TAIL: tails of the decision, by default that of the training rows)
READOUTS = TAIL_READOUTS if TAIL_HPO and READOUT_ORDER == 1 else list(dict.fromkeys(
    readout_of(points, decision, int(tail), int(decision_tail) if decision_tail else None)
    for points in os.environ.get("HAG_CANARY_POINTS", "last").split(",")
    for decision in os.environ.get("HAG_CANARY_DECISION", "").split(",")
    for tail in os.environ.get("HAG_CANARY_TAIL", "10").split(",")
    for decision_tail in os.environ.get("HAG_CANARY_DECISION_TAIL", "").split(",")))
if HPO_READOUT:  # (the readout of the studies, bachue.hpo.readout_window_study)
    points, decision, *tail = HPO_READOUT.split(":")
    READOUTS = [readout_of(points, decision, int(tail[0]) if tail else None)]
if READOUT_ORDER > 1 and any(points in ("tail", "all") for points, *_ in READOUTS):
    raise ValueError("The generalized readout needs a single row per point (HAG_CANARY_POINTS last, mean or tail_mean)")
# audio after the time of a frame in its features: half a window
DELAY = canary_mfcc_config(SONGS_SAMPLING_RATE)["win_length"] / 2 / SONGS_SAMPLING_RATE


def prepare(songs, train_songs, frame):
    """
    The 13 MFCC only (without the deltas of the cached features) and, for each song, its phrases
    songs[w]["next_phrases"] (with the calls if CALLS) and the prediction points: songs[w]["points"] (frame of each
    point, -1 for the initial zero state), songs[w]["starts"] (first frame of the phrase ending at each point, -1 for
    the point before the first phrase) and songs[w]["next"] (label of the next phrase, END after the last one).
    Returns the classes (the phrase labels and END).
    """
    annotations = annotations_of_songs() if CALLS else None
    for song in songs.values():
        song["mfcc"] = np.ascontiguousarray(song["mfcc"][:, :N_STATIC])
    for wave, song in songs.items():
        phrases = song_phrases(annotations, wave, ("SIL", "TRASH")) if CALLS else [tuple(p) for p in song["phrases"]]
        song["next_phrases"] = phrases
        labels = [label for _, _, label in phrases]
        times = [phrases[0][0]] + [end for _, end, _ in phrases]  # before the 1st, after each phrase
        song["points"] = np.array([min(int(np.floor((t - DELAY) / frame)), len(song["mfcc"]) - 1) for t in times])
        song["points"] = np.maximum(song["points"], -1)
        song["starts"] = np.array([-1] + [int(np.floor(start / frame)) for start, _, _ in phrases])
        song["next"] = np.array(labels + [END], dtype=object)
    return np.array(sorted({label for song in songs.values() for label in song["next"]} - {END}) + [END], dtype=object)


def point_windows(reservoirs, songs, waves):
    """
    Yields (wave, windows): for each prediction point of the song, the states (frames, units) of the reservoirs
    (inputs without reservoir) on the frames of the phrase ending there, up to the point (from the first frame of the
    song for the point before the first phrase); the state at the point alone when there is no such frame (the zero
    state for the initial point, the state at the point for a phrase shorter than the delay of the points). With the
    reservoirs reset at each phrase (RESET): the states of a run from a zero state on these frames (the zero state
    without frame, as for the point before the first phrase).
    """
    if not RESET:
        for part, part_states in state_chunks(reservoirs, songs, waves):
            for x, w in zip(part_states, part):
                x = np.vstack([np.zeros((1, x.shape[1]), dtype=np.float64), np.asarray(x, dtype=np.float64)])
                # (row 0: the initial zero state, point -1)
                yield w, [x[max(start, 0) + 1:point + 2] if max(start, 0) <= point else x[point + 1:point + 2]
                          for start, point in zip(songs[w]["starts"], songs[w]["points"])]
        return
    for i in range(0, len(waves), CHUNK):
        part = waves[i:i + CHUNK]
        runs = states(reservoirs, [songs[w]["mfcc"][start:point + 1] for w in part
                                   for start, point in zip(songs[w]["starts"], songs[w]["points"])
                                   if 0 <= start <= point])
        zero = np.zeros((1, runs[0].shape[1]))
        runs = iter(runs)
        for w in part:
            yield w, [np.asarray(next(runs), dtype=np.float64) if 0 <= start <= point else zero
                      for start, point in zip(songs[w]["starts"], songs[w]["points"])]


def decision_of(readout):
    """(decision, tail) of a readout: the states of its decision."""
    points, decision, tail, decision_tail = readout
    return decision, decision_tail if decision == "tail_mean" else None


def training_of(readout):
    """(training rows, tail) of a readout: its training data."""
    points, decision, tail, decision_tail = readout
    return points, tail if points in TAIL_MODES else None


def point_data(reservoirs, songs, waves, readouts, fit=(), classes=None):
    """
    States of the prediction points of the songs for the decisions of the readouts, {decision_of(readout): {wave:
    (points, units)}}: in the window of each point (point_windows), its last state or the mean of its last tail ones.
    With the songs fit (and the classes), also the training data of the readouts on them,
    {training_of(readout): (rows, labels)}: the rows of the windows with their one-hot next labels.
    """
    fit = set(fit)
    states = {decision: {} for decision in {decision_of(readout) for readout in readouts}}
    accumulated = {training: ([], []) for training in {training_of(readout) for readout in readouts}} if fit else {}
    for w, windows in point_windows(reservoirs, songs, waves):
        def selection(mode, tail):  # (the states of each window used)
            return ([x[-1:] for x in windows] if mode == "last" else windows if mode in WHOLE_MODES
                    else [x[-tail:] for x in windows])
        for (decision, tail), result in states.items():
            result[w] = np.array([x.mean(axis=0) for x in selection(decision, tail)])
        if w not in fit:
            continue
        next_labels = (songs[w]["next"][:, None] == classes[None, :]).astype(np.float64)
        for (points, tail), (inputs, labels) in accumulated.items():
            rows = selection(points, tail)
            if points in ("tail", "all"):
                inputs.append(np.vstack(rows))
                labels.append(np.repeat(next_labels, [len(x) for x in rows], axis=0))
            else:
                inputs.append(np.array([x.mean(axis=0) for x in rows]))
                labels.append(next_labels)
    if not fit:
        return states
    return states, {training: (np.vstack(inputs), np.vstack(labels))
                    for training, (inputs, labels) in accumulated.items()}


def point_states(reservoirs, songs, waves, fit=(), classes=None):
    """{wave: states of the prediction points of the songs, (points, units)} for the decision of the first readout
    (READOUTS) and, with the songs fit (and the classes), its training data on them (point_data)."""
    result = point_data(reservoirs, songs, waves, READOUTS[:1], fit, classes)
    if not fit:
        return result[decision_of(READOUTS[0])]
    return result[0][decision_of(READOUTS[0])], result[1][training_of(READOUTS[0])]


def song_data(reservoirs, songs, waves, readout, classes, frame_songs=()):
    """({wave: states of the decision of its points}, {wave: (training rows, one-hot next labels)}, moments) of the
    readout (as bachue.tasks.next_phrase.point_data, by song: the rows of a song in its time order);
    moments: (number, sum (N,), Gram matrix (N, N)) of the states of all the frames of the windows of the songs
    frame_songs (the selection "pca_frames"), None without frame_songs."""
    (points, tail), (decision, decision_tail) = training_of(readout), decision_of(readout)
    states, rows, frame_songs = {}, {}, set(frame_songs)
    n, total, gram = 0, 0.0, 0.0
    for w, windows in point_windows(reservoirs, songs, waves):
        if w in frame_songs:
            frames = np.vstack(windows).astype(np.float64)
            n, total, gram = n + len(frames), total + frames.sum(axis=0), gram + frames.T @ frames
        def selection(mode, length):
            return [x[-1:] for x in windows] if mode == "last" else [x[-length:] for x in windows]
        states[w] = np.array([x.mean(axis=0) for x in selection(decision, decision_tail)])
        next_labels = (songs[w]["next"][:, None] == classes[None, :]).astype(np.float64)
        selected = selection(points, tail)
        if points == "tail":
            rows[w] = (np.vstack(selected), np.repeat(next_labels, [len(x) for x in selected], axis=0))
        else:
            rows[w] = (np.array([x.mean(axis=0) for x in selected]), next_labels)
    return states, rows, (n, total, gram) if frame_songs else None


def reservoir_data(reservoirs, songs, waves, classes, frame_songs):
    """({wave: last state of the window of each point (points, N)}, {wave: one-hot next labels}, moments (number, sum,
    Gram matrix) of the states of all the frames of the windows of the songs frame_songs)."""
    X, Y, frame_songs = {}, {}, set(frame_songs)
    n, total, gram = 0, 0.0, 0.0
    for w, windows in point_windows(reservoirs, songs, waves):
        X[w] = np.array([x[-1] for x in windows], dtype=np.float64)
        Y[w] = (songs[w]["next"][:, None] == classes[None, :]).astype(np.float64)
        if w in frame_songs:
            frames = np.vstack(windows).astype(np.float64)
            n, total, gram = n + len(frames), total + frames.sum(axis=0), gram + frames.T @ frames
    return X, Y, (n, total, gram)


def stacked(X, Y, waves):
    """(states, labels, index of the song of each row) of the songs waves."""
    return (np.vstack([X[w] for w in waves]), np.vstack([Y[w] for w in waves]),
            np.concatenate([np.full(len(X[w]), i) for i, w in enumerate(waves)]))


def metrics(scores, true, classes):
    """Maxime's metrics of the scores (points, classes) of the next labels true (points,)."""
    predicted = classes[scores.argmax(axis=1)]
    phrase = true != END
    no_end = classes[:-1][scores[:, :-1].argmax(axis=1)]  # (END is the last class)

    def f1(t, p):
        return f1_score(t.astype(str), p.astype(str), average="macro", zero_division=0)
    return dict(accuracy=np.mean(predicted == true), f1_macro=f1(true, predicted),
                accuracy_skip_last=np.mean(predicted[phrase] == true[phrase]),
                f1_macro_skip_last=f1(true[phrase], predicted[phrase]),
                accuracy_no_end=np.mean(no_end[phrase] == true[phrase]), f1_macro_no_end=f1(true[phrase], no_end[phrase]),
                end_recall=np.mean(predicted[~phrase] == END))


# ------------------------------------------------ reservoirs ------------------------------------------------------
def readout_name(readout):
    """Suffix of the results of a single readout ("" for "last" -> "last")."""
    points, decision, tail, decision_tail = readout
    default = readout_of(points, None, tail)[1]
    return ((f"_points_{points}{tail if points in TAIL_MODES else ''}" if points != "last" else "")
            + (f"_decision_{decision}{decision_tail if decision == 'tail_mean' and points not in TAIL_MODES else ''}"
               if decision != default else "")
            + (f"_decision_tail{decision_tail}" if decision == default == "tail_mean" and decision_tail != tail else ""))
