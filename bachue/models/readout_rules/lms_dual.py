"""
NLMS readout on the monomials of the state up to degree order (the features of the generalized readout of Ohkubo &
Inubushi 2024, generalized_readout of this folder: 125 750 of them at 500 neurons for order 2), trained in its dual
form, the rule "nlms:<n>": the weights are a combination sum_j phi(x_j) A[j] of the features of the training rows, the
output of a row is K[i] @ A + b (K the kernel of the training rows: the inner products of their monomials) and the
update of the weights by row i, -eta phi(x_i) e^T, is that of its coefficient alone, A[i] -= eta e: the same rule as
LMS on the monomials themselves (bachue.models.readout_rules.lms), without computing them. The learning rate of each
row is normalized by its squared norm, the diagonal of the kernel (NLMS: the norms of the monomials spread much more
than those of the states, and a single rate diverges on the rows of the largest norms).
A pass over the rows is computed step by step, or at once as the triangular system of its steps (kernel_lms_epoch,
at_once; the course visualisations/cours/lms_forme_duale.html, section 9).

This file also holds what a dual readout is made of (KernelReadout: the centered kernel of the monomials of the
training rows, the dual weights and the outputs), which eh_dual takes from here. The learning rate of a row (row_rates)
is shared with the other readouts: bachue.models.readout_rules._utils.

Example (standardized states X of all the training rows, the order of the rows at each epoch):
    >>> readout = KernelLMS(X, order=2, learning_rate=0.1)
    >>> for epoch in range(10):
    ...     readout.partial_fit(Y, rng.permutation(len(X)))
    >>> scores = readout.run(X_test)
"""
from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy.linalg
import numpy as np

from bachue.models.readout_rules.generalized_readout import monomial_kernel
from bachue.models.readout_rules._utils import row_rates

jax.config.update("jax_enable_x64", True)  # float64, to match LMS on the monomials themselves


@partial(jax.jit, static_argnames="at_once")
def kernel_lms_epoch(A, b, K, Y, order, rates, at_once=False):
    """(A, b) after the LMS steps on the rows order of the training rows (distinct rows: a pass over them), K their
    kernel (n, n), Y their targets and rates[i] the learning rate of row i. The step of row i,

        e_i = K[i] @ A + b - Y[i],    A[i] -= rates[i] e_i,    b -= rates[i] e_i

    (the weights are sum_j phi(x_j) A[j], so the update -eta phi(x_i) e^T of the weights by row i is that of its
    coefficient alone), computed in one of two ways, the same coefficients up to rounding (about 1e-15):
      - at_once=False (default): the loop over the steps, in a compiled lax.scan;
      - at_once=True: the whole pass at once. Each step reads the changes D of the coefficients of the rows before it,
        through the kernel K + 1 (the bias receives the same changes), so in the order of the pass the steps are the
        lower triangular system

            (diag(1 / rates) + L) D = -(K @ A + b - Y),    L the part of K + 1 below the diagonal,

        solved in one call. The same n^2 m multiplications as the loop and about as fast once compiled (0.37 s against
        0.41 s per pass on 10 698 rows and 29 outputs), with one more n x n array (the kernel in the order of the pass:
        916 MB on marron1)."""
    if not at_once:
        def step(carry, i):
            A, b = carry
            error = K[i] @ A + b - Y[i]
            return (A.at[i].add(-rates[i] * error), b - rates[i] * error), None
        return jax.lax.scan(step, (A, b), order)[0]
    errors = K[order] @ A + b - Y[order]
    system = (K[order[:, None], order[None, :]] + 1.0).at[jnp.diag_indices(len(order))].set(1.0 / rates[order])
    changes = jax.scipy.linalg.solve_triangular(system, -errors, lower=True)
    return A.at[order].add(changes), b + changes.sum(axis=0)

class KernelReadout:
    """The centered kernel of the monomials of the training rows X (n, N) up to degree order (the features of the
    generalized readout of Ohkubo & Inubushi 2024: their inner products, monomial_kernel of
    generalized_readout, without computing them; centered with their mean on the training rows, as the dual
    GeneralizedRidge), and the dual weights (A, bias) of a readout trained on it by a subclass: the weights are
    sum_j phi(x_j) A[j], the output of a row K[i] @ A + bias."""

    def __init__(self, X, order):
        self.X, self.order = np.asarray(X, dtype=np.float64), order
        K = monomial_kernel(self.X, self.X, order)
        self.column_means, self.mean = K.mean(axis=0), K.mean()
        K += self.mean - self.column_means[:, None] - self.column_means[None]
        self.K = jnp.asarray(K)
        self.A, self.bias = None, None

    def kernel(self, X):
        """Centered kernel (len(X), n) of the rows X with the training rows."""
        K = monomial_kernel(np.asarray(X, dtype=np.float64), self.X, self.order)
        return K + self.mean - K.mean(axis=1, keepdims=True) - self.column_means[None]

    def run(self, X):
        return self.kernel(X) @ np.asarray(self.A) + np.asarray(self.bias)

    @property
    def Wout(self):  # (the dual weights: finite when the weights are)
        return self.A


class KernelLMS(KernelReadout):
    """LMS on the monomials, normalized (NLMS: the learning rate of row i learning_rate / (||phi(x_i)||^2 + 1)), the
    rule "nlms:<n>": the kernel Adaline of Friess & Harrison (1999, the same steps, repeated passes and the bias, with a
    constant rate), also the kernel LMS of Liu, Pokharel & Principe (2008), normalized as the kernel NLMS of Richard,
    Bermudez & Honeine (2009), without their dictionary (all the training rows are kept), on the exact kernel of the
    unique monomials of the generalized readout of Ohkubo & Inubushi (2024) (not the polynomial kernel (1 + x.y)^n,
    which weights the monomials differently): the same weights as NLMS on the monomials themselves. As an online
    readout of a reservoir: ours (none found published)."""

    def __init__(self, X, order, learning_rate, at_once=False):
        super().__init__(X, order)
        self.rates = row_rates(learning_rate, jnp.diag(self.K), True)
        self.at_once = at_once  # (each pass as one triangular system: kernel_lms_epoch)

    def partial_fit(self, Y, order):
        """Trained further on the rows order of the training rows, their targets Y."""
        if self.A is None:
            self.A, self.bias = jnp.zeros((len(self.X), Y.shape[1])), jnp.zeros(Y.shape[1])
        self.A, self.bias = kernel_lms_epoch(self.A, self.bias, self.K, jnp.asarray(Y), jnp.asarray(order), self.rates,
                                              self.at_once)
        return self
