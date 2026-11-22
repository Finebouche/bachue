"""
Settings of the tasks and of the studies, read from the variables of the environment (BACHUE_*, function env): the task and its
variants (leak, calls, reset), the readout or the rule the reservoirs are optimized for, and the prefix of the names
of the studies and of the results (STUDY_PREFIX). Their meaning: the studies of bachue.hpo (readout_window_study,
learning_rules_study, partial_order2_study).

Dataset (BACHUE_DATASET): "marron1" (default, the canary songs: the next-phrase prediction, or the phrase
boundaries) or a classification dataset of hag ("JapaneseVowels", ...: bachue.datasets.benchmarks, the task
"classification", the last state of each instance). The studies of a dataset are in one database for each kind of
study (STUDY_KIND: readout_window, learning_rules, expansions, or phrase_boundaries), outputs/<kind>/hpo/<dataset>_db.sqlite3
(DATABASE_NAME), as in hag.
"""
import os
import re

# Former names of the variables of the environment (HAG_CANARY_*, the names of canopy, until 2026-11-22): read when the
# new one is not set, as long as jobs submitted with them run (slurm/env_names.sh, the same pairs, for the scripts)
RENAMED = {
    "BACHUE_DATASET": "HAG_CANARY_DATASET",
    "BACHUE_TASK": "HAG_CANARY_TASK",
    "BACHUE_LEAKY_RESERVOIRS": "HAG_CANARY_LEAK",
    "BACHUE_WITH_CALLS": "HAG_CANARY_CALLS",
    "BACHUE_RESET_EACH_PHRASE": "HAG_CANARY_RESET",
    "BACHUE_SEARCH_READOUT": "HAG_CANARY_TAIL_HPO",
    "BACHUE_STUDY_RULE": "HAG_CANARY_HPO_RULE",
    "BACHUE_STUDY_READOUT": "HAG_CANARY_HPO_READOUT",
    "BACHUE_STUDY_MAX_EPOCHS": "HAG_CANARY_HPO_RULE_MAX_EPOCHS",
    "BACHUE_KERNEL": "HAG_CANARY_KERNEL",
    "BACHUE_N_TRIALS": "HAG_CANARY_N_TRIALS",
    "BACHUE_WORKERS": "HAG_CANARY_WORKERS",
    "BACHUE_WORKER_MEM_GB": "HAG_CANARY_WORKER_MEM",
    "BACHUE_WORKER_MEM_GB_ALL_ROWS": "HAG_CANARY_WORKER_MEM_ALL",
    "BACHUE_JOB_STUDIES_DIR": "HAG_CANARY_STUDIES",
    "BACHUE_JOB_MODE": "HAG_CANARY_MODE",
    "BACHUE_TASK_FILE": "HAG_CANARY_TASKS",
    "BACHUE_EVAL_RULES": "HAG_CANARY_READOUT_RULES",
    "BACHUE_EVAL_RULES_SUFFIX": "HAG_CANARY_RULES_SUFFIX",
    "BACHUE_ONLINE_MAX_EPOCHS": "HAG_CANARY_RULES_MAX_EPOCHS",
    "BACHUE_RULE_PROCESSES": "HAG_CANARY_RULES_JOBS",
    "BACHUE_BENCHMARK_RULES": "HAG_CANARY_BENCHMARK_RULES",
    "BACHUE_SELECTION_METHODS": "HAG_CANARY_SELECTION_METHODS",
    "BACHUE_SELECTION_RULES": "HAG_CANARY_SELECTION_RULES",
    "BACHUE_SELECTION_SIZES": "HAG_CANARY_SELECTION_SIZES",
    "BACHUE_SELECTION_SUFFIX": "HAG_CANARY_SELECTION_SUFFIX",
    "BACHUE_READOUT_ORDER": "HAG_CANARY_READOUT_ORDER",
    "BACHUE_RIDGE_SOLVER": "HAG_CANARY_RIDGE_SOLVER",
    "BACHUE_READOUT_TRAINING_ROWS": "HAG_CANARY_POINTS",
    "BACHUE_READOUT_DECISION": "HAG_CANARY_DECISION",
    "BACHUE_READOUT_TAIL": "HAG_CANARY_TAIL",
    "BACHUE_READOUT_DECISION_TAIL": "HAG_CANARY_DECISION_TAIL",
}


def env(name, default=None):
    """Value of the variable of the environment name, or of its former name (RENAMED) when name is not set, else
    default."""
    value = os.environ.get(name)
    if value is None and name in RENAMED:
        value = os.environ.get(RENAMED[name])
    return default if value is None else value


DATASET = env("BACHUE_DATASET", "marron1")
SONGS = DATASET == "marron1"  # (the canary songs; otherwise a classification dataset of hag)
# "boundaries" or "next_phrase" on the songs, "classification" on the other datasets
TASK = env("BACHUE_TASK", "boundaries") if SONGS else "classification"
LEAK = TASK != "boundaries" and env("BACHUE_LEAKY_RESERVOIRS") == "1"
# next-phrase prediction: calls among the phrases, reservoirs reset at each phrase (bachue.tasks.next_phrase)
CALLS = TASK == "next_phrase" and env("BACHUE_WITH_CALLS") == "1"
RESET = TASK == "next_phrase" and env("BACHUE_RESET_EACH_PHRASE") == "1"
# next-phrase prediction: readout (length of the tail, means in training and decision) in the search
TAIL_HPO = TASK == "next_phrase" and env("BACHUE_SEARCH_READOUT") in ("1", "2")
TAIL_SEPARATE = TAIL_HPO and env("BACHUE_SEARCH_READOUT") == "2"  # (a tail for the training, one for the decision)
N_FOLDS = 3  # (cross-validation by song on the train songs)
MAX_TAIL = 10
# next-phrase prediction: readout trained by an online rule (bachue.performances.learning_rules_evaluation), not ridge
HPO_RULE = env("BACHUE_STUDY_RULE") if TASK in ("next_phrase", "classification") and not TAIL_SEPARATE else None
# next-phrase prediction: a fixed readout of the studies, "<points>:<decision>[:<tail>]" (e.g. "tail_mean:tail_mean:10",
# "mean:mean", "last:tail_mean:5"; bachue.tasks.next_phrase.readout_of), instead of "last" -> "last"
HPO_READOUT = (env("BACHUE_STUDY_READOUT") or None
               if TASK == "next_phrase" and not TAIL_HPO and not HPO_RULE else None)


# readout on a kernel of the inputs, without reservoir (bachue.hpo.kernel_study): "volterra" (bachue.models.volterra_kernel)
KERNEL = env("BACHUE_KERNEL") or None if TASK in ("next_phrase", "classification") else None


def rule_tag(label):
    """Name of a rule in the studies and the results: "<rule>", "<rule>-order<n>" for the monomials "<rule>:<n>",
    "<rule>-<selection>" for the selected products "<rule>:<selection>" ("nlms:pca:64": "nlms-pca-64")."""
    rule, _, inputs = label.partition(":")
    if not inputs:
        return rule
    return f"{rule}-order{inputs}" if inputs.isdigit() else f"{rule}-{inputs.replace(':', '-')}"


# prefix of the studies and of the results of the next-phrase prediction
# ("JapaneseVowels": "japanese_vowels", "FSDD": "fsdd", "marron1": "marron1")
DATASET_NAME = re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", DATASET).lower()
STUDY_PREFIX = ((f"{DATASET_NAME}_" if TASK == "classification" else "next_phrase_" + "calls_" * CALLS) if KERNEL else
                f"{DATASET_NAME}_" + "leak_" * LEAK + (f"{rule_tag(HPO_RULE)}_" if HPO_RULE else "")
                if TASK == "classification" else
                f"{TASK}_" + "calls_" * CALLS + "reset_" * RESET + "leak_" * LEAK + "tail_" * (TAIL_HPO and not TAIL_SEPARATE)
                + "tail2_" * TAIL_SEPARATE + (f"{rule_tag(HPO_RULE)}_" if HPO_RULE else "")
                + (f"readout-{HPO_READOUT.replace(':', '-')}_" if HPO_READOUT else "")
                if TASK != "boundaries" else "")
# kind of the studies of the variables (their database and their results: outputs/<STUDY_KIND>)
SELECTION = bool(HPO_RULE) and not HPO_RULE.partition(":")[2].isdigit() and ":" in HPO_RULE
ORDER = bool(HPO_RULE) and HPO_RULE.partition(":")[2].isdigit()  # (a rule on all the monomials of an order, "nlms:2")
# "learning_rules": the rules on the states alone; "expansions": what the readout reads beyond the states (all the
# monomials of an order, selected products, a kernel of the inputs), with the best rules
STUDY_KIND = ("phrase_boundaries" if TASK == "boundaries" else "expansions" if KERNEL or SELECTION or ORDER
              else "learning_rules" if HPO_RULE else "readout_window")
# the former kinds of the studies of "expansions" (until 2026-11-22: partial_order2, kernels, and learning_rules for
# the monomials of an order), read when their results are copied from an older checkout
FORMER_KINDS = {"expansions": ("partial_order2", "kernels", "learning_rules")}
DATABASE_NAME = f"{DATASET_NAME}_db.sqlite3"

