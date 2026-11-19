import numpy as np
import pytest

from bachue.hpo.learning_rules_study import check_no_frames
from bachue.performances.hag_benchmarks_evaluation import seed_scores
from test_monomial_selection import data


def instances(rng, n=900):
    """({instance: (state (1, N), one-hot label (1, 4))}, {instance: state}, labels): one row per instance, the
    classes given by the sign of x_0 x_1 and of x_2."""
    X, Y, _, labels = data(rng, n=n, songs=n // 30)
    return {i: (X[i:i + 1], Y[i:i + 1]) for i in range(n)}, {i: X[i:i + 1] for i in range(n)}, labels


@pytest.mark.parametrize("rule, hyper", [("ridge", dict(ridge=1.0, epochs=None)),
                                         ("nlms", dict(normalization="standardize", rate=0.3, epochs=30)),
                                         ("rls", dict(normalization="standardize", alpha=1e-2, epochs=1))])
def test_rules_on_instances(rule, hyper):
    # the rule on the states alone stays near 0.5; with the product which gives the classes, it learns them
    rng = np.random.default_rng(0)
    train_rows, train_states, _ = instances(rng)
    _, test_states, true = instances(rng, n=300)
    linear = seed_scores(rule, hyper, train_rows, train_states, test_states, true, 0)
    selected = {key: value for key, value in hyper.items() if key != "normalization"}
    product = seed_scores(f"{rule}:residual:1", selected, train_rows, train_states, test_states, true, 0)
    assert linear["accuracy"] < 0.6 < product["accuracy"] and 0 <= product["f1_macro"] <= 1


def test_selections_without_frames():
    check_no_frames(["ridge", "nlms:pca:8"])
    with pytest.raises(ValueError):
        check_no_frames(["nlms:pca_frames:8"])
