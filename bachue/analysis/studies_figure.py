"""
Figure of the studies of the rules on selected products (bachue.hpo.partial_order2_study):
outputs/expansions/figures/selections_hpo.html, the best cross-validation accuracy of each study by selection and
by rule, for a dataset, a reservoir, a size and a leak chosen in the page; a study with less than N_TRIALS completed
trials is drawn with a dashed outline. Its two references, for the rules RULES of the page:
  - "linear": the rule on the states alone (the studies "<rule>" of bachue.hpo.learning_rules_study, and for ridge
    those of the readout "last" -> "last" of bachue.hpo.readout_window_study);
  - "order2": the rule on all the monomials of order 2 (the studies "<rule>:2" of learning_rules_study);
  - "volterra": the ridge readout on the Volterra kernel of the MFCC, without reservoir (bachue.hpo.kernel_study, kind
    "expansions"; model "none", size the window of frames: the same point for all the reservoirs and leaks of the page).

The data of the page (its block <script id="data">) are read from the databases of the studies,
outputs/<kind>/hpo/<dataset>_db.sqlite3 (kinds expansions, learning_rules and readout_window, and the former kinds
partial_order2 and kernels of expansions), and from the parts
of the jobs not merged yet (outputs/<kind>/hpo/parts/<dataset>/<study>/<dataset>_db.sqlite3); the copy of a study
with the most completed trials is kept. The test accuracies (the choice "test" of the page) are the final evaluations
of the studies, 10 seeds each, read from outputs/<kind>/performances (read_tests): <prefix>[leak_]<rule tag>_readout_rules.csv
of expansions and learning_rules (and their _parts not merged yet), <prefix>[leak_]hpo.csv of readout_window (the
linear ridge readout), <prefix><kernel>_<window>.csv of expansions; the evaluations without a rule tag (all the rules on
the reservoirs of another study) are not read. From the root of a checkout:
    python -m bachue.analysis.studies_figure            # reads the databases, updates the page
    python -m bachue.analysis.studies_figure json       # prints the data (on PlaFRIM, where the parts are)
    python -m bachue.analysis.studies_figure <file>     # updates the page from these printed data and the databases
"""
import csv
import json
import math
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

from bachue.common import OUTPUTS_DIR
from bachue.hpo.studies import N_TRIALS
from bachue.models.reservoirs import PURE_MODELS

# (and the former kinds of expansions, partial_order2 and kernels, read when copied from an older checkout)
KINDS = ("expansions", "learning_rules", "readout_window", "partial_order2", "kernels")
RULES = ("ridge", "nlms", "eh_np_local", "neh")  # (the rules of the page, among those of the references)
FIGURE = OUTPUTS_DIR / "expansions" / "figures" / "selections_hpo.html"
# rows of the cross-validation of each dataset (the prediction points of the train songs, the train instances: the standard
# error of an accuracy in the page) and number of outputs (the products of the selection "scores")
ROWS = {"marron1": 10698, "japanese_vowels": 270}  # (marron1: one row per prediction point, calls and the point before
# the first phrase included, BACHUE_WITH_CALLS=1)
TEST_ROWS = {"japanese_vowels": 370}  # (test instances: the binomial standard error of a test accuracy in the page)
OUTPUTS = {"marron1": 29, "japanese_vowels": 9}
# "[next_phrase_calls_|<dataset>_][leak_]<rule>-<selection>_<model>_<size>" (bachue.settings.STUDY_PREFIX, rule_tag),
# and the studies on the states alone, "[leak_][<rule>_]<model>_<size>" (without rule: the ridge readout)
MODEL = "|".join(PURE_MODELS)
SELECTED = re.compile(rf"^(?P<leak>leak_)?(?P<rule>[^-]+)-(?P<selection>.+)_(?P<model>{MODEL})_(?P<size>\d+)$")
STATES = re.compile(rf"^(?P<leak>leak_)?(?:(?P<rule>[a-z_]+?)_)?(?P<model>{MODEL})_(?P<size>\d+)$")
QUERY = """select s.study_name, count(v.value), max(v.value) from studies s
           left join trials t on t.study_id = s.study_id and t.state = 'COMPLETE'
           left join trial_values v on v.trial_id = t.trial_id group by s.study_name"""


def study_of(kind, name):
    """(leak, rule, selection, model, size) of a study of a kind shown in the page, None otherwise."""
    selected, states = SELECTED.match(name), STATES.match(name)
    order = selected and re.fullmatch(r"order\d+", selected["selection"])
    if kind in ("expansions", "learning_rules") and order:  # (all the monomials of an order; learning_rules: former)
        match, rule, selection = selected, selected["rule"], selected["selection"]
    elif kind in ("expansions", "partial_order2") and selected:
        match, rule, selection = selected, selected["rule"], selected["selection"].replace("-", ":")
    elif kind == "learning_rules" and states and states["rule"] in RULES:
        match, rule, selection = states, states["rule"], "linear"
    elif kind == "readout_window" and states and states["rule"] is None:
        match, rule, selection = states, "ridge", "linear"
    elif kind in ("expansions", "kernels") and (kernel := re.fullmatch(r"(?P<kernel>[a-z]+)_(?P<window>\d+)", name)):
        # (a ridge readout on a kernel of the inputs, without reservoir: model "none", size its window of frames)
        return [0, "ridge", kernel["kernel"], "none", int(kernel["window"])]
    else:
        return None
    return [int(bool(match["leak"])), rule, selection, match["model"], int(match["size"])]


def read_studies(known=()):
    """dict(date, n_trials, rows, outputs, studies: [[dataset, leak, rule, selection, model, size, best accuracy,
    completed trials]]) of the databases and of the parts of the kinds KINDS, and of the studies known (read
    elsewhere)."""
    studies = {tuple(study[:6]): study for study in known}
    for kind in KINDS:
        folder = OUTPUTS_DIR / kind / "hpo"
        for database in sorted(folder.glob("*.sqlite3")) + sorted(folder.glob("parts/*/*/*.sqlite3")):
            dataset = database.name.removesuffix("_db.sqlite3")
            connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)  # (read-only: no file created)
            try:
                rows = list(connection.execute(QUERY))
            except sqlite3.DatabaseError as error:  # (e.g. the pointer of a file of Git LFS not downloaded)
                print(f"{database}: not read ({error})", file=sys.stderr)
                continue
            finally:
                connection.close()
            for name, done, best in rows:
                described = study_of(kind, name.removeprefix("next_phrase_calls_").removeprefix(f"{dataset}_"))
                if described is None:
                    continue
                study = [dataset] + described + [None if best is None else round(100 * best, 2), done]
                if done >= studies.get(tuple(study[:6]), study)[7]:
                    studies[tuple(study[:6])] = study
    return dict(date=datetime.now().strftime("%Y-%m-%d %H:%M"), n_trials=N_TRIALS, rows=ROWS, outputs=OUTPUTS,
                test_rows=TEST_ROWS, studies=sorted(studies.values()))


def tested(kind, path):
    """(dataset, leak, rule, selection) of a file of final evaluations, rule and selection None when read from its
    rows, None for a file not shown."""
    stem = path.stem.removesuffix("_cv")
    if path.parent.name.endswith("_parts"):
        stem = path.parent.name.removesuffix("_parts")
    for prefix, dataset in [("next_phrase_calls_", "marron1")] + [(f"{name}_", name) for name in ROWS]:
        if stem.startswith(prefix):
            rest = stem.removeprefix(prefix)
            break
    else:
        return None
    leak = rest.startswith("leak_")
    rest = rest.removeprefix("leak_")
    if kind in ("expansions", "partial_order2", "learning_rules") and rest.endswith("_readout_rules"):
        tag = rest.removesuffix("_readout_rules")
        return (dataset, int(leak), None, None) if tag else None  # (a rule tag: the studies of a rule)
    if kind == "readout_window":
        return (dataset, int(leak), "ridge", "linear") if rest == "hpo" else None
    if kind in ("expansions", "kernels") and (kernel := re.fullmatch(r"(?P<kernel>[a-z]+)_(?P<window>\d+)", rest)):
        return (dataset, 0, "ridge", kernel["kernel"])
    return None


def read_tests(known=()):
    """[[dataset, leak, rule, selection, model, size, mean test accuracy, seeds, standard deviation over the seeds]]
    of the final evaluations of the studies (accuracies in %), and of the tests known (read elsewhere: the one with
    more seeds kept)."""
    scores = {}  # (dataset, leak, rule, selection, model, size) -> {seed: accuracy}
    for kind in KINDS:
        folder = OUTPUTS_DIR / kind / "performances"
        # (<dataset>/<file>, <dataset>/<parts>/<file>; the flat files and parts of an older checkout too)
        for path in sorted(folder.glob("*.csv")) + sorted(folder.glob("*/*.csv")) + sorted(folder.glob("*/*/*.csv")):
            if path.stem.endswith("_cv") or (described := tested(kind, path)) is None:
                continue
            dataset, leak, rule, selection = described
            with path.open() as file:
                for row in csv.DictReader(file):
                    if "condition" not in row:  # (a kernel of the inputs, without reservoir: its window)
                        model, size, seed = "none", int(row["window"]), row.get("seed", 0)
                    else:
                        model, size, seed = row["condition"], int(row["network_size"]), row["seed"]
                        if model not in PURE_MODELS:
                            continue
                    if rule is None:  # (the rule and the selection of the row: "nlms:pca:64", "ridge:2", "neh")
                        name, _, inputs = row["rule"].partition(":")
                        if name not in RULES:
                            continue
                        rule_name = name
                        chosen = row.get("selection") or inputs
                        chosen = f"order{inputs}" if inputs.isdigit() else (chosen or "linear")
                    else:
                        rule_name, chosen = rule, selection
                    key = (dataset, leak, rule_name, chosen, model, size)
                    scores.setdefault(key, {})[str(seed)] = 100 * float(row["accuracy"])
    tests = {tuple(test[:6]): test for test in known}
    for key, by_seed in scores.items():
        values = list(by_seed.values())
        mean = sum(values) / len(values)
        deviation = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1)) if len(values) > 1 else 0.0
        test = list(key) + [round(mean, 2), len(values), round(deviation, 2)]
        if test[7] >= tests.get(key, test)[7]:
            tests[key] = test
    return sorted(tests.values())


def write_figure(data):
    """Replaces the data of the page (the block <script id="data" type="application/json">)."""
    page = FIGURE.read_text()
    block = re.compile(r'(<script id="data" type="application/json">).*?(</script>)', re.DOTALL)
    if not block.search(page):
        raise ValueError(f"{FIGURE}: no block <script id=\"data\">")
    text = json.dumps(data, separators=(",", ":")).replace("],[", "],\n[")
    FIGURE.write_text(block.sub(lambda match: f"{match[1]}\n{text}\n{match[2]}", page))
    complete = sum(study[7] >= data["n_trials"] for study in data["studies"])
    print(f"{FIGURE}: {len(data['studies'])} studies, {complete} with {data['n_trials']} completed trials, "
          f"{len(data.get('tests', []))} final evaluations ({data['date']})")


if __name__ == "__main__":
    argument = sys.argv[1] if len(sys.argv) > 1 else ""
    if argument == "json":
        print(json.dumps(read_studies() | dict(tests=read_tests())))
    elif argument:  # (the data printed elsewhere, at their date, completed by the databases of this checkout)
        printed = json.loads(Path(argument).read_text())
        write_figure(read_studies(printed["studies"]) | dict(date=printed["date"],
                                                             tests=read_tests(printed.get("tests", []))))
    else:
        write_figure(read_studies() | dict(tests=read_tests()))
