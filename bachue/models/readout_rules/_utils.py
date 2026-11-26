"""
The learning rate of each row (row_rates), shared by the online readouts in JAX of this folder (lms, eh, and their dual
forms lms_dual, eh_dual): constant (LMS, EH) or divided by the squared norm of the row + 1 (NLMS, neh).
Each rule stays in its own file, with the loop over its steps and the update of its weights: the step of the
exploratory Hebbian rule (eh_step) is in eh, the centered kernel of the monomials and its dual weights (KernelReadout)
in lms_dual, and eh_dual imports both from there.
"""
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)  # float64, to match reservoirpy and the NumPy node


def row_rates(learning_rate, squared_norms, normalized):
    """Learning rate of each row: learning_rate (LMS, EH), or learning_rate / (squared norm + 1) (normalized: NLMS,
    and the step of neh)."""
    squared_norms = jnp.asarray(squared_norms)
    return learning_rate / (squared_norms + 1) if normalized else jnp.full(squared_norms.shape, learning_rate)

