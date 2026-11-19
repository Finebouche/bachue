"""
Ridge readouts fitted on rows and their labels: reservoirpy's Ridge (solver HAG_CANARY_RIDGE_SOLVER, default "svd":
the singular value decomposition of the centered states, stable with the smallest coefficients) or, with
HAG_CANARY_READOUT_ORDER=n (n >= 2), the generalized readout of order n (bachue.models.generalized_readout).
"""
import os

from reservoirpy.nodes import Ridge

from bachue.models.generalized_readout import GeneralizedRidge

READOUT_ORDER = int(os.environ.get("HAG_CANARY_READOUT_ORDER", "1"))  # 1: linear ridge readout, n: generalized readout
RIDGE_SOLVER = os.environ.get("HAG_CANARY_RIDGE_SOLVER", "svd")  # (of reservoirpy.nodes.Ridge, linear readout)


def fitted_readouts(data, ridges, order=None):
    """Yields (ridge, the readout trained on the data (rows, labels) with this coefficient) for each coefficient of
    ridges: reservoirpy's Ridge (solver RIDGE_SOLVER), or the generalized readout of order (default READOUT_ORDER) > 1
    (a single fit, solved again for each coefficient)."""
    order = order or READOUT_ORDER
    if order == 1:
        for ridge in ridges:
            yield ridge, Ridge(ridge=ridge, solver=RIDGE_SOLVER).fit(*data)
        return
    readout = GeneralizedRidge(order=order).fit(*data)
    for ridge in ridges:
        yield ridge, readout.set_ridge(ridge)


def fit_readout(data, ridge):
    """The readout trained on the data (rows, labels) with the coefficient ridge (fitted_readouts)."""
    return next(fitted_readouts(data, [ridge]))[1]
