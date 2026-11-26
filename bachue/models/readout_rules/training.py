"""
Training of a readout by a rule on the rows of the songs (or instances): the rule and its inputs from its label
(rule_inputs: "<rule>", "<rule>:<order>" or "<rule>:<selection>"), the normalization of the inputs (normalized), the
inputs of a selection of products (selected_rows, bachue.models.product_selection) and the readout trained for a
number of epochs (trained). The rules and their versions: bachue.models.readout_rules; their evaluation:
bachue.performances.learning_rules_evaluation.
"""
import jax.numpy as jnp
import numpy as np

from bachue.models.readout_rules.generalized_readout import GeneralizedRidge
from bachue.models.product_selection import Selection
from bachue.models.readout_rules.eh import EH
from bachue.models.readout_rules.eh_dual import KernelEH
from bachue.models.readout_rules.eh_numpy import version_options
from bachue.models.readout_rules.lms import LMS
from bachue.models.readout_rules.lms_dual import KernelLMS
from bachue.models.readout_rules.rls import RLS
from bachue.models.readout_rules.ridge_readouts import fitted_readouts

NORMALIZED_RULES = ("nlms", "neh")  # (the rules which can read the monomials of the state: order > 1, with ridge)


def normalized(hyper, rows, states, fit):
    """(rows, states) with the inputs normalized as hyper["normalization"] with the statistics of the rows of the songs
    fit (unchanged without normalization: ridge)."""
    if "normalization" not in hyper:
        return rows, states
    X = np.vstack([rows[w][0] for w in fit])
    mean = X.mean(axis=0)
    scale = X.std(axis=0) + 1e-8 if hyper["normalization"] == "standardize" else 1.0
    return ({w: ((x - mean) / scale, y) for w, (x, y) in rows.items()},
            {w: (x - mean) / scale for w, x in states.items()})


def rule_inputs(label):
    """(rule, order, selection) of a rule of BACHUE_EVAL_RULES: "<rule>" (order 1, the states), "<rule>:<order>"
    (the monomials of the states up to degree order: ridge and NORMALIZED_RULES) or "<rule>:<selection>" (order 1, the
    states and the products of a selection of monomial_selection, e.g. "nlms:pca:64"; selection None otherwise)."""
    rule, _, inputs = label.partition(":")
    if inputs and not inputs.isdigit():
        return rule, 1, inputs
    order = int(inputs or 1)
    if order > 1 and rule != "ridge" and rule not in NORMALIZED_RULES:
        raise ValueError(f"{label}: only ridge and {', '.join(NORMALIZED_RULES)} read the monomials (order > 1)")
    return rule, order, None


def rule_order(label):
    """(rule, order) of a rule (rule_inputs)."""
    return rule_inputs(label)[:2]


_selected = {}  # (the inputs of the last selection built: its key and themselves)


def selected_rows(selection, all_rows, fit, moments, seed):
    """({wave: (inputs of its rows, labels)} of the songs fit, function of the states giving their inputs) of a
    selection of monomial_selection chosen on the rows of the songs fit: the standardized states and products."""
    X, Y = (np.vstack([all_rows[w][i] for w in fit]) for i in (0, 1))
    key = (selection, seed, X.shape, float(X.sum()), float(X[:, 0] @ X[:, -1]))
    if _selected.get("key") != key:
        lengths = [len(all_rows[w][0]) for w in fit]
        groups = np.repeat(np.arange(len(fit)), lengths)
        F, inputs = Selection(X, Y, groups, moments, None, seed=seed).features(selection)
        bounds = np.cumsum([0] + lengths)
        _selected.update(key=key, value=({w: (F[bounds[i]:bounds[i + 1]], all_rows[w][1]) for i, w in enumerate(fit)}, inputs))
    return _selected["value"]


_generalized = {}  # (the last generalized ridge readout fitted, solved again for each coefficient: its key and itself)


def trained(rule, hyper, rows, fit, rng, epochs, order=1, selected=False):
    """Yields (epochs, readout) of the rule trained on the rows of the songs fit (order > 1: on the monomials of the
    rows up to degree order): the ridge readout (epochs None), or the online readout after each number of epochs of
    epochs (the same node, trained further), until it diverges (non-finite weights: stopped after the next number of
    epochs, its scores NaN). selected: the rows of a selection (ridge: a single fit for all the coefficients)."""
    if rule == "ridge":
        data = (np.vstack([rows[w][0] for w in fit]), np.vstack([rows[w][1] for w in fit]))
        if order == 1 and not selected:
            yield None, next(fitted_readouts(data, [hyper["ridge"]], order))[1]
            return
        key = (order, data[0].shape, float(data[0].sum()), float(data[0][:, 0] @ data[0][:, -1]))
        if _generalized.get("key") != key:  # (a single fit for all the coefficients of the same rows)
            _generalized.update(key=key, readout=GeneralizedRidge(order=order).fit(*data))
        yield None, _generalized["readout"].set_ridge(hyper["ridge"])
        return
    mean_norm = np.mean(np.concatenate([np.sum(rows[w][0] ** 2, axis=1) for w in fit])) + 1  # (with the bias)
    # (JAX rules: the rows of all the songs fit, the songs in the order of rng at each epoch, as one pass over them)
    X, Y = (np.vstack([rows[w][i] for w in fit]) for i in (0, 1))
    if order > 1:  # (the dual form, on the kernel of the monomials of the rows, always normalized)
        if rule == "nlms":
            node = KernelLMS(X, order, hyper["rate"])
        else:
            options = {k: v for k, v in version_options(rule).items() if k != "normalized"}
            node = KernelEH(X, order, hyper["rate"], noise=hyper["noise"], seed=rng, **options)
    elif rule == "rls":
        node = RLS(alpha=hyper["alpha"])
    elif rule in ("lms", "nlms"):
        node = LMS(learning_rate=hyper["rate"] / mean_norm) if rule == "lms" else LMS(hyper["rate"], normalized=True)
    else:  # (the versions of EH; the rate of the unnormalized ones divided by the mean squared norm, as lms)
        options = version_options(rule)
        rate = hyper["rate"] if options["normalized"] else hyper["rate"] / mean_norm
        node = EH(learning_rate=rate, noise=hyper["noise"], seed=rng, **options)
    X, Y = jnp.asarray(X), jnp.asarray(Y)
    starts = np.cumsum([0] + [len(rows[w][0]) for w in fit])
    with np.errstate(all="ignore"):  # (diverging rates: their scores are NaN, never chosen)
        for epoch in range(1, max(epochs) + 1):
            if rule == "rls":
                for w in rng.permutation(fit):
                    node.partial_fit(*rows[w])
            else:
                order_rows = np.concatenate([np.arange(starts[s], starts[s + 1]) for s in rng.permutation(len(fit))])
                if order > 1:
                    node.partial_fit(Y, order_rows)
                else:
                    node.partial_fit(X, Y, order_rows)
            if epoch in epochs:
                yield epoch, node
                if not np.all(np.isfinite(node.Wout)):
                    return
