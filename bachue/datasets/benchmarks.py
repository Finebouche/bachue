"""
Datasets of hag (load_*, pipeline, ...: copied from hag.datasets on 2026-11-19), to study the readouts on other tasks
than the canary songs: the benchmarks of HAG's evaluations, prepared as there (bachue.datasets.pipeline.prepare_data: spectral representation of
the univariate data, MinMax scaling fitted on the train set, pretraining instances of the HAG learning).

Three kinds of tasks (bachue.datasets.load_data):
  - "forecasting": a time series and its value step_ahead steps later (MackeyGlass, Lorenz, Henon, NARMA,
    Sunspot_daily); train and test are arrays (T, features), Y_train and Y_test arrays (T, outputs);
  - "classification": a label for each instance (JapaneseVowels, FSDD, SPEECHCOMMANDS, SpokenArabicDigits, CatsDogs,
    and the wingbeats of insects FruitFlies, InsectSound and MosquitoSound);
    train and test are lists of arrays (T_i, features), Y_train and Y_test one-hot labels;
  - "categorical_forecasting": the category of the next element of a sequence, handled as a classification (Canary:
    the audio of a phrase and the label of the next one).
The "Canary" of hag is one instance per phrase (its audio); the studies of bachue read the whole songs instead
(bachue.datasets.load_categorical_forecasting.load_songs: MFCC without floor; bachue.common.load_dataset). Both have
the corrected annotations of bachue/datasets/datasets/marron1.

The spectral representation is that of HAG's test evaluations (hag.performances.run_test_evaluation): "mfcc" for the
audio and already spectral datasets (MFCC_DATASETS), "stft" for the others. Some datasets are downloaded into
bachue/datasets/datasets at their first use (JapaneseVowels, FSDD, SPEECHCOMMANDS, Canary, the aeon ones); the forecasting ones are generated.

The classification datasets as rows of a readout: instance_data (the last state of the reservoirs on each instance).

Example:
    >>> data = load("MackeyGlass")
    >>> data.kind, data.train.shape, data.Y_train.shape
"""
from dataclasses import dataclass

import numpy as np

from bachue.datasets.load_data import CATEGORICAL_FORECASTING_DATASETS, CLASSIFICATION_DATASETS, FORECASTING_DATASETS
from bachue.datasets.pipeline import prepare_data
from bachue.models.reservoirs import states

KINDS = {**{name: "forecasting" for name in FORECASTING_DATASETS},
         **{name: "classification" for name in CLASSIFICATION_DATASETS},
         **{name: "categorical_forecasting" for name in CATEGORICAL_FORECASTING_DATASETS}}
# datasets read through their MFCC in HAG's test evaluations (SPECTRAL_DATASETS of hag.performances.run_test_evaluation,
# not imported: that module loads the recurrent networks of the comparisons)
MFCC_DATASETS = {"CatsDogs", "JapaneseVowels", "FSDD", "SpokenArabicDigits", "SPEECHCOMMANDS", "Canary", "FruitFlies", "InsectSound", "MosquitoSound"}
SEED = 0  # (of the pretraining instances, as hag.performances.run_test_evaluation)


@dataclass(frozen=True)
class Benchmark:
    """A dataset of hag prepared for a reservoir: pretrain (instances of the HAG learning), train and test (inputs),
    Y_train and Y_test (targets), see the module docstring for their shapes."""
    name: str
    kind: str
    pretrain: object
    train: object
    test: object
    Y_train: object
    Y_test: object
    is_multivariate: bool
    is_instances_classification: bool


def names(kind=None):
    """Names of the datasets of hag, sorted (kind: those of a kind of KINDS only)."""
    return sorted(name for name, k in KINDS.items() if kind is None or k == kind)


def spectral_representation_of(name):
    """Spectral representation of a dataset in HAG's test evaluations: "mfcc" or "stft"."""
    return "mfcc" if name in MFCC_DATASETS else "stft"


def load(name, spectral_representation=None, step_ahead=5, data_type="normal", noise_std=0.001, seed=SEED):
    """The dataset name of hag, prepared by bachue.datasets.pipeline.prepare_data (spectral_representation: "stft",
    "mfcc" or "none", default spectral_representation_of(name); step_ahead: horizon of the forecasting
    tasks; data_type "noisy": with noise of standard deviation noise_std). seed: of numpy's global generator, which
    draws the pretraining instances of the classification datasets (None: left as it is)."""
    if name not in KINDS:
        raise ValueError(f"Unknown dataset {name!r}: expected one of {names()}")
    if seed is not None:
        np.random.seed(seed)
    prepared = prepare_data(name, spectral_representation or spectral_representation_of(name), data_type=data_type,
                            noise_std=noise_std, step_ahead=step_ahead)
    return Benchmark(name, KINDS[name], *prepared)


def instance_data(reservoirs, inputs, Y):
    """({instance: its last state (1, N)}, {instance: (its last state (1, N), its one-hot label (1, classes))}): each
    instance as a song of a single row (the songs of readout_rules), the reservoirs run from a zero state."""
    last = [np.asarray(x[-1:], dtype=np.float64) for x in states(reservoirs, list(inputs))]
    Y = np.asarray(Y, dtype=np.float64)
    return {i: x for i, x in enumerate(last)}, {i: (x, Y[i:i + 1]) for i, x in enumerate(last)}
