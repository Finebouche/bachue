"""
Phrase boundaries of the canary songs (bachue.performances.phrase_boundaries_evaluation): boundary targets, peak picking and scores;
the songs with their boundary targets; the ridge coefficient and the peak threshold chosen by cross-validation
(cross_validate). (Their studies, those of canopy: not optimized in bachue.)
"""
import numpy as np
from scipy.signal import find_peaks
from sklearn.model_selection import KFold

from bachue.common import RIDGES, load_dataset as load_songs_dataset, pretrain_phrases
from bachue.hpo.studies import CV_SEED, FOLD_SEED
from bachue.models.reservoirs import build, normal_equations, readout_outputs, ridge_readout
from bachue.settings import N_FOLDS

BUMP_STD = 0.025  # s
TOLERANCE = 0.05  # s
MIN_PEAK_DISTANCE = 0.1  # s
THRESHOLDS = np.linspace(0.05, 0.95, 19)


def scores(outputs, songs, waves, threshold, frame):
    """
    Scores of the boundaries detected (peaks of the outputs above threshold) in the songs. Predicted and true times are
    matched one to one within TOLERANCE, closest pairs first: when the nearest predictions of the true times within
    TOLERANCE are all different, each of them gets its nearest prediction (computed with searchsorted); otherwise the
    pairs are sorted by distance (slower).
    """
    nb_predicted, matched, silence = 0, [], []
    for output, wave in zip(outputs, waves):
        peaks, _ = find_peaks(output, height=threshold, distance=max(1, round(MIN_PEAK_DISTANCE / frame)))
        predicted = np.sort(peaks * frame)
        true = np.array([t for t, _ in songs[wave]["boundaries"]], dtype=float)
        nb_predicted += len(predicted)
        song_matched = np.zeros(len(true), dtype=bool)
        if len(predicted) and len(true):
            index = np.searchsorted(predicted, true)
            left, right = np.clip(index - 1, 0, len(predicted) - 1), np.clip(index, 0, len(predicted) - 1)
            left_distance, right_distance = np.abs(true - predicted[left]), np.abs(true - predicted[right])
            nearest = np.where(left_distance <= right_distance, left, right)
            song_matched = np.minimum(left_distance, right_distance) <= TOLERANCE
            if len(np.unique(nearest[song_matched])) < song_matched.sum():
                pairs = sorted((abs(p - t), i, j) for i, p in enumerate(predicted) for j, t in enumerate(true)
                               if abs(p - t) <= TOLERANCE)
                used_predicted, song_matched = set(), np.zeros(len(true), dtype=bool)
                for _, i, j in pairs:
                    if i not in used_predicted and not song_matched[j]:
                        used_predicted.add(i)
                        song_matched[j] = True
        matched.append(song_matched)
        silence.append(np.array([s for _, s in songs[wave]["boundaries"]], dtype=bool))
    matched, silence = np.concatenate(matched), np.concatenate(silence)
    precision = matched.sum() / max(nb_predicted, 1)
    recall = matched.mean() if len(matched) else 0.0
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return dict(f1=f1, precision=precision, recall=recall,
                recall_silence=matched[silence].mean() if silence.any() else np.nan,
                recall_no_silence=matched[~silence].mean() if (~silence).any() else np.nan)


def load_dataset():
    """bachue.common.load_dataset, and the boundary target of each song: songs[wave]["target"]."""
    songs, train_songs, test_songs, frame = load_songs_dataset()
    for song in songs.values():
        times = np.arange(len(song["mfcc"])) * frame  # librosa frames are centered
        song["target"] = np.zeros(len(times))
        for time, _ in song["boundaries"]:
            song["target"] = np.maximum(song["target"], np.exp(-0.5 * ((times - time) / BUMP_STD) ** 2))
    return songs, train_songs, test_songs, frame


def cross_validate(model, params, size, songs, train_songs, frame):
    """(out-of-fold F1, ridge, threshold) of the model on the train songs: in each fold of a 3-fold cross-validation by
    song, the reservoirs are built (HAG learning on phrases of the fold's training songs), the ridge readout is
    trained on the fold's training songs for every coefficient of RIDGES and outputs the boundaries of the fold's
    validation songs; the ridge coefficient and the peak threshold are then chosen on the out-of-fold outputs of all
    the train songs."""
    folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED).split(train_songs)
    outputs = {}  # (ridge index, wave) -> readout output
    for k, (fit_index, val_index) in enumerate(folds):
        fit = [train_songs[i] for i in fit_index]
        val = [train_songs[i] for i in val_index]
        np.random.seed(CV_SEED + k)
        reservoirs = build(model, params, size, pretrain_phrases(songs, fit, frame))
        statistics = normal_equations(reservoirs, songs, fit)
        Wouts = [ridge_readout(statistics, ridge) for ridge in RIDGES]
        for w, output in readout_outputs(reservoirs, songs, val, Wouts).items():
            for r in range(len(RIDGES)):
                outputs[r, w] = output[:, r]

    best = (-1.0, None, None)
    for r, ridge in enumerate(RIDGES):
        song_outputs = [outputs[r, w] for w in train_songs]
        for threshold in THRESHOLDS:
            f1 = scores(song_outputs, songs, train_songs, threshold, frame)["f1"]
            if f1 > best[0]:
                best = (f1, float(ridge), float(threshold))
    return best
