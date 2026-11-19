"""
Shared parts of the three studies of the reservoirs (bachue.hpo.readout_window_study, bachue.hpo.learning_rules_study,
bachue.hpo.partial_order2_study): the search space of the reservoirs (reservoir_parameters), the run of a study by
several workers (run_study), the storage of the studies, their names (study_name_of: the prefix of bachue.settings,
e.g. "next_phrase_calls_leak_nlms_random_ee_500", "japanese_vowels_leak_nlms-pca-64_random_ee_500") and their best
trial (best_trial); and the seeds of the cross-validation, shared with the cross-validations of the readouts (its
number of folds: bachue.settings.N_FOLDS).

Storage, as in hag: all the studies of a kind and of a dataset in one SQLite database,
outputs/<kind>/hpo/<dataset>_db.sqlite3 (bachue.settings.STUDY_KIND, DATABASE_NAME; e.g.
outputs/partial_order2/hpo/marron1_db.sqlite3). Several jobs writing the same SQLite file would corrupt it: a job of a study
(slurm/canary.sbatch) optimizes it in a database of its own, in HAG_CANARY_STUDIES (in /dev/shm during the job, copied
back to its part, outputs/<kind>/hpo/parts/<dataset>/<study>/: python -m bachue.hpo.studies part <model> <size>), started
from the study of the database of the dataset if there is one; the parts are merged into the databases of the datasets
afterwards (python -m bachue.hpo.studies merge [kind]). The databases of canopy (one file per study) are imported
with python -m bachue.hpo.studies import <folder>.

The search space of the reservoirs (the same ranges as hag.hpo.hpo_esn), for each part of the model (prefix "<part>:"
in a hybrid): input scaling and bias scaling; with HAG_CANARY_LEAK=1, the leak rate (1 otherwise); the HAG learning
of "mean_hag" and "var_hag" (weight increment, instances, increments, target rate or variance and intrinsic
plasticity); the connectivity and the spectral radius of the ESN. The hybrids are not optimized: they combine the
parts optimized alone (bachue.performances.readout_window_evaluation.conditions).
"""
import os
import re
import sys
from pathlib import Path

import optuna
from joblib import Parallel, delayed
from optuna.samplers import TPESampler
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState

from bachue.common import OUTPUTS_DIR, load_dataset
from bachue.models.reservoirs import ESN_MODELS, HAG_MODELS, PURE_MODELS, parts
from bachue.settings import DATABASE_NAME, DATASET, DATASET_NAME, LEAK, STUDY_KIND, STUDY_PREFIX, TAIL_HPO, TASK

# the database of the studies of the variables: that of the dataset (DATABASE), or that of a job (HAG_CANARY_STUDIES)
DATABASE = OUTPUTS_DIR / STUDY_KIND / "hpo" / DATABASE_NAME
JOB_STUDIES = os.environ.get("HAG_CANARY_STUDIES")
STUDIES_DATABASE = Path(JOB_STUDIES) / DATABASE_NAME if JOB_STUDIES else DATABASE
FOLD_SEED = 0
CV_SEED = 1000  # matrices of fold k drawn with np.random.seed(CV_SEED + k), the same for all the trials
N_TRIALS = int(os.environ.get("HAG_CANARY_N_TRIALS", 400))


def storage(study_name=None, database=None):
    """Storage of the studies (default: of STUDIES_DATABASE; all the studies of a kind and of a dataset share it)."""
    database = Path(database or STUDIES_DATABASE)
    database.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{database}"


def study_names(database=None):
    """Names of the studies of a database (none if it does not exist: opening a missing SQLite database creates it)."""
    database = Path(database or STUDIES_DATABASE)
    if not database.exists():
        return set()
    return set(optuna.get_all_study_names(storage(database=database)))


def study_exists(study_name):
    """Whether the study is in its database."""
    return study_name in study_names()


def stop_after_failures(max_failures=10):
    """Callback stopping the optimization after max_failures failed trials in a row (systematic error)."""
    def callback(study, trial):
        last = [t.state for t in study.get_trials(deepcopy=False)][-max_failures:]
        if len(last) == max_failures and all(state == TrialState.FAIL for state in last):
            study.stop()
    return callback


def study_name_of(model, size):
    return f"{STUDY_PREFIX}{model}_{size}"


def best_trial(study_name):
    if not study_exists(study_name):  # (without creating an empty database)
        return None
    try:
        return optuna.load_study(study_name=study_name, storage=storage(study_name)).best_trial
    except (KeyError, ValueError):
        return None


def reservoir_parameters(trial, model, length):
    """Parameters of the reservoirs of a trial (see the module docstring); length: number of frames of the longest
    phrase of the train songs (range of the increments of the HAG learning)."""
    params = {}
    for part, p in parts(model):
        params[p + "input_scaling"] = trial.suggest_float(p + "input_scaling", 0.01, 0.2, step=0.005)
        if LEAK:
            params[p + "leaky_rate"] = trial.suggest_float(p + "leaky_rate", 0.01, 1, log=True)
        params[p + "bias_scaling"] = trial.suggest_float(p + "bias_scaling", 0, 0.2, step=0.005)
        if part in HAG_MODELS:
            params[p + "weight_increment"] = trial.suggest_float(p + "weight_increment", 0.001, 0.1, step=0.001)
            params[p + "use_full_instance"] = trial.suggest_categorical(p + "use_full_instance", [True, False])
            params[p + "min_increment"] = trial.suggest_int(p + "min_increment", 3, length)
            params[p + "max_increment"] = trial.suggest_int(p + "max_increment", params[p + "min_increment"], length * 5)
            if part == "mean_hag":
                params[p + "target_rate"] = trial.suggest_float(p + "target_rate", 0.5, 1, step=0.01)
                params[p + "rate_spread"] = trial.suggest_float(p + "rate_spread", 0.01, 0.4, step=0.005)
            else:
                params[p + "variance_target"] = trial.suggest_float(p + "variance_target", 0.001, 0.02, step=0.001)
                params[p + "variance_spread"] = trial.suggest_float(p + "variance_spread", 0.001, 0.05, step=0.002)
                params[p + "intrinsic_saturation"] = trial.suggest_float(p + "intrinsic_saturation", 0.8, 0.98, step=0.02)
                params[p + "intrinsic_coef"] = trial.suggest_float(p + "intrinsic_coef", 0.8, 0.98, step=0.02)
        elif part in ESN_MODELS:
            params[p + "connectivity"] = trial.suggest_float(p + "connectivity", 0, 1)
            params[p + "spectral_radius"] = trial.suggest_float(p + "spectral_radius", 0.4, 1.6, step=0.01)
        else:
            raise ValueError(f"Unknown model {part}")
    return params


def next_phrase_data():
    """(songs, train songs, frame, classes of the next phrase, frames of the longest phrase of the train songs) of the
    studies on the songs: the MFCC without the floor of the dB relative to the maximum of the whole song (it depends on
    the future)."""
    from bachue.tasks.next_phrase import prepare
    songs, train_songs, _, frame = load_dataset(top_db=None)
    classes = prepare(songs, train_songs, frame)
    length = max(int(end / frame) + 1 - int(start / frame) for w in train_songs for start, end, _ in songs[w]["phrases"])
    return songs, train_songs, frame, classes, length


def instances_data():
    """(the classification dataset DATASET of hag, prepared: bachue.datasets.benchmarks.load, frames of its longest
    train instance) of the studies on the other datasets."""
    from bachue.datasets.benchmarks import load
    data = load(DATASET)
    return data, max(len(x) for x in data.train)


def _worker(model, size, objective_of, worker_index):
    loaded = next_phrase_data() if TASK == "next_phrase" else instances_data()
    study_name = study_name_of(model, size)
    study = optuna.load_study(study_name=study_name, storage=storage(study_name),
                              sampler=TPESampler(constant_liar=True, seed=worker_index))
    study.optimize(objective_of(model, size, *loaded), gc_after_trial=True, catch=(Exception,),
                   callbacks=[MaxTrialsCallback(N_TRIALS, states=(TrialState.COMPLETE,)), stop_after_failures()])


def run_study(model, size, objective_of, readout_of_trial, tasks=("next_phrase",)):
    """Optimizes the study of the model (N_TRIALS complete trials, HAG_CANARY_WORKERS processes, default 1, each with
    its own TPE sampler) and prints its best trial. objective_of(model, size, *data): the objective of a study, data
    those of next_phrase_data (the task "next_phrase") or of instances_data ("classification"); readout_of_trial(trial):
    the description of its readout; tasks: those of the study. In a job (HAG_CANARY_STUDIES), the study starts from
    that of the database of the dataset if it is there and not in the database of the job."""
    if TASK not in tasks or TAIL_HPO:
        raise ValueError(f"The task {TASK} (dataset {DATASET}): the studies of this kind are those of the tasks "
                         f"{tasks} (HAG_CANARY_TASK, HAG_CANARY_DATASET), with a fixed readout (HAG_CANARY_TAIL_HPO=0)")
    if model not in PURE_MODELS:
        raise ValueError(f"Unknown model {model}, expected one of {PURE_MODELS} (the hybrids <hag>+<esn> are not "
                         "optimized: they combine the parts optimized alone, see bachue.performances.readout_window_evaluation.conditions)")
    n_workers = int(os.environ.get("HAG_CANARY_WORKERS", 1))
    study_name = study_name_of(model, size)
    if JOB_STUDIES and study_name not in study_names() and study_name in study_names(DATABASE):
        optuna.copy_study(from_study_name=study_name, from_storage=storage(database=DATABASE), to_storage=storage())
        print(f"{study_name}: resumed from {DATABASE}", flush=True)
    study = optuna.create_study(study_name=study_name, storage=storage(study_name), direction="maximize",
                                load_if_exists=True)
    done = len([t for t in study.trials if t.state == TrialState.COMPLETE])
    print(f"{study_name}: {done} / {N_TRIALS} trials done, {n_workers} workers, {STUDIES_DATABASE}", flush=True)
    if done < N_TRIALS:
        Parallel(n_jobs=n_workers)(delayed(_worker)(model, size, objective_of, i) for i in range(n_workers))
    best = optuna.load_study(study_name=study_name, storage=storage(study_name)).best_trial
    print(f"best trial {best.number}: accuracy {best.value:.4f}, {readout_of_trial(best)}, {best.params}")


# ---- the parts of the jobs and the databases of the datasets
def completed(study):
    return sum(t.state == TrialState.COMPLETE for t in study.trials)


def copy_into(name, source, target):
    """Copies the study name of the database source into the database target, replacing it if source has more
    completed trials. Returns a description."""
    source_study = optuna.load_study(study_name=name, storage=storage(database=source))
    if name in study_names(target):
        current = optuna.load_study(study_name=name, storage=storage(database=target))
        if completed(current) >= completed(source_study):
            return f"{name}: kept ({completed(current)} completed trials >= {completed(source_study)})"
        optuna.delete_study(study_name=name, storage=storage(database=target))
    optuna.copy_study(from_study_name=name, from_storage=storage(database=source), to_storage=storage(database=target))
    return f"{name}: {completed(source_study)} completed trials copied into {target}"


def merge(kinds=None):
    """Merges the parts of the jobs (outputs/<kind>/hpo/parts/<dataset>/<study>/<dataset>_db.sqlite3) into the databases
    of the datasets (outputs/<kind>/hpo/<dataset>_db.sqlite3)."""
    kinds = kinds or sorted(path.name for path in OUTPUTS_DIR.iterdir() if (path / "hpo" / "parts").is_dir())
    for kind in kinds:
        for part in sorted((OUTPUTS_DIR / kind / "hpo" / "parts").glob("*/*/*.sqlite3")):
            for name in sorted(study_names(part)):
                print(copy_into(name, part, OUTPUTS_DIR / kind / "hpo" / part.name))


def kind_of(study_name):
    """(kind, dataset name) of a study of canopy (one database per study, named by the study)."""
    if not study_name.startswith("next_phrase_"):
        return "phrase_boundaries", "marron1"
    rest = re.sub(r"^next_phrase_((calls|reset|leak|tail|tail2)_)*", "", study_name)
    if rest.startswith("readout-") or any(rest.startswith(f"{model}_") for model in PURE_MODELS):
        return "readout_window", "marron1"
    tag = rest.split("_", 1)[0] if "-" in rest.split("_", 1)[0] else re.split(r"_(?=" + "|".join(PURE_MODELS) + ")", rest)[0]
    if "-" in tag and "-order" not in tag:
        return "partial_order2", "marron1"
    return "learning_rules", "marron1"


def import_canopy(folder):
    """Copies the studies of canopy (folder/<study>.sqlite3, one study each) into the databases of the datasets."""
    for path in sorted(Path(folder).glob("*.sqlite3")):
        kind, dataset = kind_of(path.stem)
        for name in sorted(study_names(path)):
            print(copy_into(name, path, OUTPUTS_DIR / kind / "hpo" / f"{dataset}_db.sqlite3"))


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "part":  # (the folder of the part of a job, slurm/canary.sbatch)
        print(OUTPUTS_DIR / STUDY_KIND / "hpo" / "parts" / DATASET_NAME / study_name_of(sys.argv[2], int(sys.argv[3])))
    elif command == "merge":
        merge(sys.argv[2:] or None)
    elif command == "import":
        import_canopy(sys.argv[2])
    else:
        raise ValueError(f"Unknown command {command!r}: part <model> <size>, merge [kinds], import <folder>")
