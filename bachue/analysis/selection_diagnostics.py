"""
Diagnostics of the selections of products (bachue.models.product_selection), from the trained readout on all the
products of z ("full_std"): the importance of each product, sum over the outputs of its squared weight times its
variance; the share of the total importance of the M most important products ("top_pairs"), of the products of each
selection of units or pairs ("share"), and the effective number of parameters of "full_std" ("effective_parameters":
sum of mu / (mu + ridge) over the eigenvalues mu of the centered kernel).
"""
import numpy as np

from bachue.models.product_selection import PAIR_KINDS, SIZES, UNIT_KINDS

TOP_PAIRS = [m * (m + 1) // 2 for m in SIZES] + [50000]  # (diagnostics)


def selection_diagnostics(selection, full_std, methods):
    """[dict(measure, method, size, value)] of a Selection, from the readout "full_std" trained on its rows (see
    the module docstring)."""
    A, n = full_std.dual_coef, selection.Z.shape[1]
    importance = np.zeros((n, n))
    for k in range(A.shape[1]):  # (weight of z_i z_j for the output k: sum over the rows t of A[t, k] z_ti z_tj)
        importance += ((selection.Z * A[:, k:k + 1]).T @ selection.Z) ** 2
    importance *= selection.pair_variance
    values = np.sort(importance[np.triu_indices(n)])[::-1]
    total = values.sum()
    rows = [dict(measure="top_pairs", method="", size=m, value=values[:m].sum() / total) for m in TOP_PAIRS]
    for method in methods:
        if method.partition(":")[0] in UNIT_KINDS + PAIR_KINDS:
            i, j = selection.unit_pairs(method)
            rows.append(dict(measure="share", method=method, size=len(i), value=importance[i, j].sum() / total))
    mu = full_std.eigenvalues
    rows.append(dict(measure="effective_parameters", method="full_std", size=len(values),
                     value=float(np.sum(mu / (mu + full_std.ridge)))))
    return rows
