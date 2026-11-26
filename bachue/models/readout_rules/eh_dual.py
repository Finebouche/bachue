"""
Exploratory Hebbian readout on the monomials of the state up to degree order (the features of the generalized readout
of Ohkubo & Inubushi 2024, generalized_readout of this folder), trained in its dual form, the rule "neh:<n>": the
weights are a combination sum_j phi(x_j) A[j] of the features of the training rows, the output of a row is K[i] @ A + b
(K the kernel of the training rows: the inner products of their monomials) and the update of the weights by row i,
phi(x_i) exploration^T, is that of its coefficient alone, A[i] += exploration: the same rule as EH on the monomials
themselves (bachue.models.readout_rules.eh), without computing them. The learning rate of each row is normalized by
its squared norm, the diagonal of the kernel (as NLMS: the norms of the monomials spread much more than those of the
states, and a single rate diverges on the rows of the largest norms).

The step of the rule (eh_step: baseline, reward, modulation, learning rate) is that of
bachue.models.readout_rules.eh, and the centered kernel of the monomials with its dual weights and its outputs
(KernelReadout) that of bachue.models.readout_rules.lms_dual. The learning rate of a row (row_rates) is shared with the
other readouts: bachue.models.readout_rules._utils.

Example (standardized states X of all the training rows, the order of the rows at each epoch):
    >>> readout = KernelEH(X, order=2, learning_rate=0.03, noise=1.5, seed=0)
    >>> for epoch in range(10):
    ...     readout.partial_fit(Y, rng.permutation(len(X)))
    >>> scores = readout.run(X_test)
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from bachue.models.readout_rules._utils import row_rates
from bachue.models.readout_rules.eh import eh_step
from bachue.models.readout_rules.lms_dual import KernelReadout

jax.config.update("jax_enable_x64", True)  # float64, to match EH on the monomials themselves


@partial(jax.jit, static_argnames=("baseline", "reward", "modulation", "fit_bias", "decay"))
def kernel_eh_epoch(carry, K, Y, order, noise, rates, alpha, baseline, reward, modulation, fit_bias, decay):
    """carry (A, b, z_bar, P_bar, steps) after the EH steps on the rows order of the training rows, K their kernel,
    with the exploration noise noise (len(order), output_dim) and the learning rate rates[i] of row i: the update
    phi(x_i) exploration^T of the weights by row i is that of its coefficient alone, A[i] += exploration (exploration:
    the vector of eh_step)."""
    def step(carry, inputs):
        A, b, z_bar, P_bar, steps = carry
        i, xi = inputs
        exploration, z_bar, P_bar = eh_step(K[i] @ A + b, xi, Y[i], z_bar, P_bar, steps, rates[i], alpha, baseline,
                                            reward, modulation, decay)
        A = A.at[i].add(exploration)
        if fit_bias:
            b = b + exploration
        return (A, b, z_bar, P_bar, steps + 1), None
    return jax.lax.scan(step, carry, (order, noise))[0]


class KernelEH(KernelReadout):
    """EH on the monomials (the options of bachue.models.readout_rules.eh.EH), always normalized as KernelLMS; the
    exploration noise drawn as EH (with the same generator, the same outputs as EH on the monomials themselves). The
    defaults: the version "neh" (bachue.models.readout_rules.eh_numpy.VERSIONS), the rule "neh:<n>": neh on the
    monomials, ours (no reward-modulated or perturbation rule on nonlinear features of a reservoir found published)."""

    def __init__(self, X, order, learning_rate=1e-3, decay=None, noise=0.5, baseline="sample", alpha=0.2,
                 reward="per_output", modulation="binary", fit_bias=True, seed=None):
        super().__init__(X, order)
        self.rates = row_rates(learning_rate, jnp.diag(self.K), True)
        self.decay, self.noise, self.alpha = decay, noise, alpha
        self.baseline, self.reward, self.modulation, self.fit_bias = baseline, reward, modulation, fit_bias
        self.rng = np.random.default_rng(seed)
        self.carry = None

    def partial_fit(self, Y, order):
        """Trained further on the rows order of the training rows, their targets Y."""
        n_out = Y.shape[1]
        if self.carry is None:
            self.carry = (jnp.zeros((len(self.X), n_out)), jnp.zeros(n_out), jnp.zeros(n_out), jnp.zeros(n_out), 0)
        noise = self.rng.uniform(-self.noise, self.noise, (len(order), n_out))
        self.carry = kernel_eh_epoch(self.carry, self.K, jnp.asarray(Y), jnp.asarray(order), jnp.asarray(noise),
                                     self.rates, self.alpha, baseline=self.baseline, reward=self.reward,
                                     modulation=self.modulation, fit_bias=self.fit_bias, decay=self.decay)
        self.A, self.bias = self.carry[0], self.carry[1]
        return self
