import os
import subprocess
import sys
from pathlib import Path

from bachue.settings import RENAMED, env

ROOT = Path(__file__).resolve().parents[1]


def bash_pairs():
    text = (ROOT / "slurm" / "env_names.sh").read_text()
    block = text.split('BACHUE_RENAMED="', 1)[1].split('"', 1)[0]
    return dict(line.split() for line in block.strip().splitlines())


def test_the_scripts_and_python_rename_the_same_variables():
    assert bash_pairs() == RENAMED
    assert len(RENAMED) == len(set(RENAMED.values())) == 32


def test_env_reads_the_former_name_when_the_new_one_is_not_set(monkeypatch):
    monkeypatch.delenv("BACHUE_LEAKY_RESERVOIRS", raising=False)
    monkeypatch.delenv("HAG_CANARY_LEAK", raising=False)
    assert env("BACHUE_LEAKY_RESERVOIRS", "0") == "0"
    monkeypatch.setenv("HAG_CANARY_LEAK", "1")
    assert env("BACHUE_LEAKY_RESERVOIRS", "0") == "1"
    monkeypatch.setenv("BACHUE_LEAKY_RESERVOIRS", "0")
    assert env("BACHUE_LEAKY_RESERVOIRS") == "0"  # (the new name first)


def study_prefix(variables):
    environment = {k: v for k, v in os.environ.items() if not k.startswith(("BACHUE_", "HAG_CANARY_"))}
    environment.update(variables, PYTHONPATH=str(ROOT))
    command = [sys.executable, "-c", "from bachue.settings import STUDY_PREFIX; print(STUDY_PREFIX)"]
    return subprocess.run(command, env=environment, capture_output=True, text=True, check=True).stdout.strip()


def test_the_former_names_give_the_same_studies():
    former = {"HAG_CANARY_TASK": "next_phrase", "HAG_CANARY_CALLS": "1", "HAG_CANARY_LEAK": "1",
              "HAG_CANARY_HPO_RULE": "nlms:pca:64"}
    new = {name: former[old] for name, old in RENAMED.items() if old in former}
    assert study_prefix(former) == study_prefix(new) == "next_phrase_calls_leak_nlms-pca-64_"
    assert study_prefix({"HAG_CANARY_DATASET": "JapaneseVowels"}) == "japanese_vowels_"


def test_the_scripts_read_the_former_names():
    command = 'source slurm/env_names.sh; echo "$BACHUE_LEAKY_RESERVOIRS $BACHUE_TASK_FILE ${BACHUE_KERNEL:-unset}"'
    environment = {"PATH": os.environ["PATH"], "HAG_CANARY_LEAK": "1", "HAG_CANARY_TASKS": "tasks.txt",
                   "BACHUE_TASK_FILE": "new.txt"}
    out = subprocess.run(["bash", "-c", command], cwd=ROOT, env=environment, capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["1", "new.txt", "unset"]
