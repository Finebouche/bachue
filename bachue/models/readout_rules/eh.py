"""
Exploratory Hebbian readout trained in JAX: the rule of bachue.models.readout_rules.eh_numpy.EH (see its docstring for
the rule, its four ingredients and its versions, VERSIONS: "legenstein", "eh", "eh_np", "eh_np_local", "neh"), the
steps of a pass over the rows in a compiled lax.scan, about 6 times faster than the reservoirpy node's loop over the
steps. The same options and defaults (the version "eh"), EH.version(name, ...) as the node; the exploration noise drawn
with the NumPy generator in the order of the node (one draw of output_dim values per step): with the same generator, the
same weights (float64). The same rule on the features of a kernel ("neh:<n>", ours) is in
bachue.models.readout_rules.eh_dual (KernelEH), which calls the step of the rule of this file (eh_step: baseline,
reward, modulation, learning rate): eh applies the vector of a step to the weights (dW = x exploration^T), eh_dual to
the coefficient of the row (dA[i] = exploration).

Example (rows of all the training songs, the order of the rows at each epoch):
    >>> readout = EH.version("eh_np_local", learning_rate=1e-3, noise=0.5, seed=0)
    >>> for epoch in range(10):
    ...     readout.partial_fit(X, Y, rng.permutation(len(X)))
    >>> scores = readout.run(X_test)
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from bachue.models.readout_rules._utils import row_rates
from bachue.models.readout_rules.eh_numpy import check_options, version_options

jax.config.update("jax_enable_x64", True)  # float64, to match the NumPy node


def eh_step(output, xi, y, z_bar, P_bar, steps, rate, alpha, baseline, reward, modulation, decay):
    """(exploration, z_bar, P_bar) of a step of the exploratory Hebbian rule (bachue.models.readout_rules.eh_numpy:
    its four ingredients) on a row of target y: output its output without noise, xi the exploration noise, (z_bar,
    P_bar) the baselines of the output and of the performance before the step and after it, steps the number of
    steps already done, rate the learning rate of the row; exploration is the vector to add to the outputs of the row
    (and to the bias). Called in a compiled loop: baseline, reward, modulation and decay are constants of the loop."""
    def performance(out):  # (2. reward)
        return -(out - y) ** 2 if reward == "per_output" else jnp.full(out.shape, -jnp.sum((out - y) ** 2))

    z = output + xi
    P = performance(z)
    if baseline == "sample":  # (1. baseline)
        z_bar = output
        P_bar = performance(z_bar)
    else:  # (first step: the baselines start at the first output and performance)
        z_bar, P_bar = jnp.where(steps == 0, z, z_bar), jnp.where(steps == 0, P, P_bar)
    M = (P > P_bar).astype(z.dtype) if modulation == "binary" else P - P_bar  # (3. modulation)
    eta = rate / (1 + steps / decay) if decay else rate  # (4. learning rate: rate normalized or not)
    exploration = eta * (z - z_bar) * M
    if baseline == "filtered":
        z_bar, P_bar = (1 - alpha) * z_bar + alpha * z, (1 - alpha) * P_bar + alpha * P
    return exploration, z_bar, P_bar


@partial(jax.jit, static_argnames=("baseline", "reward", "modulation", "fit_bias", "decay"))
def eh_epoch(carry, X, Y, order, noise, rates, alpha, baseline, reward, modulation, fit_bias, decay):
    """carry (W, b, z_bar, P_bar, steps) after the EH steps on the rows order of (X, Y), with the exploration noise
    noise (len(order), output_dim), the learning rate of row i rates[i]: the update of the weights by row i is
    dW = x_i exploration^T, exploration the vector of eh_step."""
    def step(carry, inputs):
        W, b, z_bar, P_bar, steps = carry
        i, xi = inputs
        x = X[i]
        exploration, z_bar, P_bar = eh_step(x @ W + b, xi, Y[i], z_bar, P_bar, steps, rates[i], alpha, baseline,
                                            reward, modulation, decay)
        W = W + jnp.outer(x, exploration)
        if fit_bias:
            b = b + exploration
        return (W, b, z_bar, P_bar, steps + 1), None
    return jax.lax.scan(step, carry, (order, noise))[0]


class EH:
    """EH readout, initial weights and bias zero (the options and defaults of bachue.models.readout_rules.eh_numpy.EH:
    the version "eh")."""

    def __init__(self, learning_rate=1e-3, decay=None, noise=0.5, baseline="filtered", alpha=0.2, reward="global",
                 modulation="binary", normalized=False, fit_bias=True, seed=None):
        check_options(baseline, reward, modulation)
        self.learning_rate, self.decay, self.noise, self.alpha = learning_rate, decay, noise, alpha
        self.baseline, self.reward, self.modulation, self.fit_bias = baseline, reward, modulation, fit_bias
        self.normalized = normalized
        self.rng = np.random.default_rng(seed)
        self.carry, self.rates = None, None

    @classmethod
    def version(cls, name, **kwargs):
        """EH of the version name of bachue.models.readout_rules.eh_numpy.VERSIONS, with the other parameters kwargs."""
        return cls(**version_options(name), **kwargs)

    def partial_fit(self, X, Y, order):
        """Trained further on the rows order of (X, Y) (arrays, JAX ones to avoid a copy at each call)."""
        if self.carry is None:
            n_out = Y.shape[1]
            self.carry = (jnp.zeros((X.shape[1], n_out)), jnp.zeros(n_out), jnp.zeros(n_out), jnp.zeros(n_out), 0)
            self.rates = row_rates(self.learning_rate, jnp.sum(jnp.asarray(X) ** 2, axis=1), self.normalized)
        noise = self.rng.uniform(-self.noise, self.noise, (len(order), Y.shape[1]))
        self.carry = eh_epoch(self.carry, X, Y, jnp.asarray(order), jnp.asarray(noise), self.rates,
                              self.alpha, baseline=self.baseline, reward=self.reward, modulation=self.modulation,
                              fit_bias=self.fit_bias, decay=self.decay)
        return self

    @property
    def Wout(self):
        return self.carry[0]

    @property
    def bias(self):
        return self.carry[1]

    def run(self, X):
        return np.asarray(X @ self.Wout + self.bias)
