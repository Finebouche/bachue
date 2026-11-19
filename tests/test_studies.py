import optuna
import pytest

from bachue.hpo.studies import completed, copy_into, kind_of, study_names


@pytest.mark.parametrize("name, kind", [
    ("next_phrase_calls_random_ee_500", "readout_window"),
    ("next_phrase_calls_leak_readout-tail_mean-tail_mean-10_var_hag_1000", "readout_window"),
    ("next_phrase_calls_leak_eh_np_local_random_ei_500", "learning_rules"),
    ("next_phrase_calls_neh-order2_mean_hag_1000", "learning_rules"),
    ("next_phrase_calls_nlms-pca-64_random_ee_500", "partial_order2"),
    ("next_phrase_calls_ridge-random_pairs-2080_var_hag_500", "partial_order2"),
    ("random_ee_500", "phrase_boundaries")])
def test_kind_of_the_studies_of_canopy(name, kind):
    assert kind_of(name) == (kind, "marron1")


def study(path, name, values):
    storage = f"sqlite:///{path}"
    s = optuna.create_study(study_name=name, storage=storage, direction="maximize", load_if_exists=True)
    for value in values:
        s.add_trial(optuna.trial.create_trial(params={"x": value}, distributions={"x": optuna.distributions.FloatDistribution(0, 1)}, value=value))


def test_copy_into_keeps_the_study_with_the_most_trials(tmp_path):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    part, target = tmp_path / "part.sqlite3", tmp_path / "target.sqlite3"
    study(part, "a", [0.1, 0.2])
    study(target, "b", [0.5])
    copy_into("a", part, target)  # (a new study: copied)
    assert study_names(target) == {"a", "b"}
    study(part, "b", [0.3])  # (fewer trials than the database of the dataset: kept)
    assert "kept" in copy_into("b", part, target)
    study(part, "b", [0.4, 0.6])  # (more trials: replaced)
    copy_into("b", part, target)
    assert completed(optuna.load_study(study_name="b", storage=f"sqlite:///{target}")) == 3
