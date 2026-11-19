"""
Settings of the tasks and of the studies, read from the variables of the environment (HAG_CANARY_*): the task and its
variants (leak, calls, reset), the readout or the rule the reservoirs are optimized for, and the prefix of the names
of the studies and of the results (STUDY_PREFIX). Their meaning: the studies of bachue.hpo (readout_window_study,
learning_rules_study, partial_order2_study).

Dataset (HAG_CANARY_DATASET): "marron1" (default, the canary songs: the next-phrase prediction, or the phrase
boundaries) or a classification dataset of hag ("JapaneseVowels", ...: bachue.datasets.benchmarks, the task
"classification", the last state of each instance). The studies of a dataset are in one database for each kind of
study (STUDY_KIND: readout_window, learning_rules, partial_order2, or phrase_boundaries), outputs/<kind>/hpo/<dataset>_db.sqlite3
(DATABASE_NAME), as in hag.
"""
import os
import re

DATASET = os.environ.get("HAG_CANARY_DATASET", "marron1")
SONGS = DATASET == "marron1"  # (the canary songs; otherwise a classification dataset of hag)
# "boundaries" or "next_phrase" on the songs, "classification" on the other datasets
TASK = os.environ.get("HAG_CANARY_TASK", "boundaries") if SONGS else "classification"
LEAK = TASK != "boundaries" and os.environ.get("HAG_CANARY_LEAK") == "1"
# next-phrase prediction: calls among the phrases, reservoirs reset at each phrase (bachue.tasks.next_phrase)
CALLS = TASK == "next_phrase" and os.environ.get("HAG_CANARY_CALLS") == "1"
RESET = TASK == "next_phrase" and os.environ.get("HAG_CANARY_RESET") == "1"
# next-phrase prediction: readout (length of the tail, means in training and decision) in the search
TAIL_HPO = TASK == "next_phrase" and os.environ.get("HAG_CANARY_TAIL_HPO") in ("1", "2")
TAIL_SEPARATE = TAIL_HPO and os.environ.get("HAG_CANARY_TAIL_HPO") == "2"  # (a tail for the training, one for the decision)
N_FOLDS = 3  # (cross-validation by song on the train songs)
MAX_TAIL = 10
# next-phrase prediction: readout trained by an online rule (bachue.performances.learning_rules_evaluation), not ridge
HPO_RULE = os.environ.get("HAG_CANARY_HPO_RULE") if TASK in ("next_phrase", "classification") and not TAIL_SEPARATE else None
# next-phrase prediction: a fixed readout of the studies, "<points>:<decision>[:<tail>]" (e.g. "tail_mean:tail_mean:10",
# "mean:mean", "last:tail_mean:5"; bachue.tasks.next_phrase.readout_of), instead of "last" -> "last"
HPO_READOUT = (os.environ.get("HAG_CANARY_HPO_READOUT") or None
               if TASK == "next_phrase" and not TAIL_HPO and not HPO_RULE else None)


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
STUDY_PREFIX = (f"{DATASET_NAME}_" + "leak_" * LEAK + (f"{rule_tag(HPO_RULE)}_" if HPO_RULE else "")
                if TASK == "classification" else
                f"{TASK}_" + "calls_" * CALLS + "reset_" * RESET + "leak_" * LEAK + "tail_" * (TAIL_HPO and not TAIL_SEPARATE)
                + "tail2_" * TAIL_SEPARATE + (f"{rule_tag(HPO_RULE)}_" if HPO_RULE else "")
                + (f"readout-{HPO_READOUT.replace(':', '-')}_" if HPO_READOUT else "")
                if TASK != "boundaries" else "")
# kind of the studies of the variables (their database and their results: outputs/<STUDY_KIND>)
SELECTION = bool(HPO_RULE) and not HPO_RULE.partition(":")[2].isdigit() and ":" in HPO_RULE
STUDY_KIND = ("phrase_boundaries" if TASK == "boundaries" else "partial_order2" if SELECTION
              else "learning_rules" if HPO_RULE else "readout_window")
DATABASE_NAME = f"{DATASET_NAME}_db.sqlite3"

