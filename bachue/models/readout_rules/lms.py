"""
Online least mean squares readout (the rule of reservoirpy's LMS, the LMS-based FORCE rule of Sussillo & Abbott 2009)
trained in JAX: the steps of a pass over the rows in a compiled lax.scan, about 6 times faster than reservoirpy's loop
over the steps (500 inputs, 30 outputs), the same weights (float64):

    e = W^T x + b - y,    dW = -eta x e^T,    db = -eta e

The two versions differ only by the learning rate eta (rules of bachue.performances.learning_rules_evaluation):
  - LMS (Widrow & Hoff 1960, "Adaptive switching circuits"), normalized=False, rule "lms": a single eta for all the
    rows and all the steps (reservoirpy's LMS). readout_rules gives it as eta = c / (mean ||x||^2 + 1), the mean
    computed once on the training rows (the classical step c / tr(R) of Widrow and Haykin, stable in the mean for c < 2
    whatever the size of the reservoir): still LMS, the same algorithm, only its step written as a fraction c of the
    stability bound, so that the same grid of c fits every reservoir.
  - NLMS (normalized LMS: Nagumo & Noda 1967, Albert & Gardner 1967), normalized=True, rule "nlms": eta recomputed for
    each row from its own squared norm (+ 1, the input of the bias), eta = learning_rate / (||x||^2 + 1) (epsilon-NLMS,
    epsilon = 1), stable for a learning rate below 2 whatever the norms of the rows. In reservoir computing, the same
    step is in the BPDC rule of Steil (2004, as a decorrelation, with a backpropagated error term), and an NLMS
    readout of an echo state network is evaluated by Yuenyong (2014, PhD thesis, Tokyo Tech: worse than RLS).
The same rule on the features of a kernel, written on the coefficients of the training rows ("nlms:<n>"), is in
bachue.models.readout_rules.lms_dual (KernelLMS). The learning rate of each row (row_rates) is shared with it and
with eh: bachue.models.readout_rules._utils.

Example (rows of all the training songs, the order of the rows at each epoch):
    >>> readout = LMS(learning_rate=1e-3)
    >>> for epoch in range(10):
    ...     readout.partial_fit(X, Y, rng.permutation(len(X)))
    >>> scores = readout.run(X_test)
"""
import jax
import jax.numpy as jnp
import numpy as np

from bachue.models.readout_rules._utils import row_rates

jax.config.update("jax_enable_x64", True)  # float64, to match reservoirpy


@jax.jit
def lms_epoch(W, b, X, Y, order, rates):
    """(W, b) after the LMS steps on the rows order of (X, Y), the learning rate of row i rates[i]."""
    def step(carry, i):
        W, b = carry
        error = X[i] @ W + b - Y[i]
        return (W - rates[i] * jnp.outer(X[i], error), b - rates[i] * error), None
    return jax.lax.scan(step, (W, b), order)[0]


class LMS:
    """LMS readout, initial weights and bias zero (reservoirpy's LMS with its defaults): LMS (Widrow & Hoff), a constant
    learning rate; normalized: NLMS, the learning rate of each row divided by its squared norm + 1 (see the module
    docstring)."""

    def __init__(self, learning_rate, normalized=False):
        self.learning_rate, self.normalized = learning_rate, normalized
        self.Wout, self.bias, self.rates = None, None, None

    def partial_fit(self, X, Y, order):
        """Trained further on the rows order of (X, Y) (arrays, JAX ones to avoid a copy at each call)."""
        if self.Wout is None:
            self.Wout, self.bias = jnp.zeros((X.shape[1], Y.shape[1])), jnp.zeros(Y.shape[1])
            self.rates = row_rates(self.learning_rate, jnp.sum(jnp.asarray(X) ** 2, axis=1), self.normalized)
        self.Wout, self.bias = lms_epoch(self.Wout, self.bias, X, Y, jnp.asarray(order), self.rates)
        return self

    def run(self, X):
        return np.asarray(X @ self.Wout + self.bias)
