"""
Volterra reservoir kernel of Gonon, Grigoryeva & Ortega (2022), "Reservoir kernels and Volterra series"
(https://arxiv.org/abs/2212.14641): the inner product of the features of an infinite reservoir, the state-space
representation of the Volterra series, on two input histories z = (z_0, z_-1, ...) and z' (z_0 the current frame,
the frames before the start of a sequence equal to zero). Its features are all the monomials of all the degrees of
the frames at all the delays, a monomial of degree i of a frame weighted by tau^i and the delay j by lambda^j; their
inner product has a closed form (their equation (14)),

    K(z, z') = 1 + sum_{k >= 1} lambda^(2k) prod_{j=0}^{k-1} 1 / (1 - tau^2 <z_-j, z'_-j>),

computed from the oldest delay to the current frame: K = 1 + lambda^2 K / (1 - tau^2 <z_-j, z'_-j>) (their recursion
(15)), started from 1 / (1 - lambda^2), its value on a history of zero frames. The histories are cut at a window of W
frames (lagged): the frames older than W are taken as zero. Conditions of the paper: tau ||z|| < 1 for all the
frames and lambda^2 < 1 - tau^2 M^2 (M the largest norm of a frame); with tau = theta / M and lambda = mu
sqrt(1 - theta^2), theta and mu in [0, 1[, every pair is valid (parameters), the frames of norm above M (outside the
train data) brought back to the norm M (clipped).

Example (the last frame of each instance as the prediction point, window of 32 frames):
    >>> A, B = lagged_instances(train, 32), lagged_instances(test, 32)
    >>> norm = max_norm(train)
    >>> tau, lam = parameters(0.5, 0.9, norm)
    >>> K, K_test = volterra_gram(A, A, tau, lam), volterra_gram(clipped(B, norm), A, tau, lam)
"""
import numpy as np

BLOCK = 2048  # rows of the kernel computed at once (memory)


def lagged_instances(sequences, window):
    """Histories (n, window, d) of the last frame of each sequence (frames, d): row i, lag u is the frame u steps
    before the last one, zero before the first frame."""
    sequences = [np.asarray(x, dtype=np.float64) for x in sequences]
    out = np.zeros((len(sequences), window, sequences[0].shape[1]))
    for i, x in enumerate(sequences):
        recent = x[::-1][:window]
        out[i, :len(recent)] = recent
    return out


def lagged_points(sequence, points, window):
    """Histories (len(points), window, d) of the frames points of a sequence (frames, d) (-1: before the first frame,
    a history of zeros): lag u is the frame point - u, zero before the first frame."""
    sequence = np.asarray(sequence, dtype=np.float64)
    padded = np.vstack([np.zeros((window, sequence.shape[1])), sequence])
    points = np.asarray(points)
    index = points[:, None] + window - np.arange(window)[None]  # (frame point - u in the padded sequence)
    return padded[index]


def max_norm(histories):
    """Largest norm of a frame of the histories (n, window, d)."""
    return float(np.linalg.norm(histories, axis=2).max())


def clipped(histories, norm):
    """The histories with the frames of norm above norm brought back to the norm norm."""
    norms = np.linalg.norm(histories, axis=2, keepdims=True)
    return histories * np.minimum(1.0, norm / np.maximum(norms, 1e-300))


def parameters(theta, mu, norm):
    """(tau, lambda) of theta and mu in [0, 1[ for frames of norm at most norm: tau = theta / norm, lambda = mu
    sqrt(1 - theta^2) (the conditions of the paper: tau norm < 1 and lambda^2 < 1 - tau^2 norm^2)."""
    if not (0 <= theta < 1 and 0 <= mu < 1):
        raise ValueError(f"theta={theta}, mu={mu}: both in [0, 1[")
    return theta / norm, mu * np.sqrt(1 - theta ** 2)


def volterra_gram(A, B, tau, lam):
    """Volterra kernel (len(A), len(B)) of the histories A (n, W, d) and B (m, W, d) (lag 0: the current frame)."""
    symmetric = A is B  # (the kernel of rows with themselves: the blocks above the diagonal only, then mirrored)
    A, B = np.asarray(A, dtype=np.float64), np.asarray(B, dtype=np.float64)
    if A.shape[1:] != B.shape[1:]:
        raise ValueError(f"Histories of shapes {A.shape} and {B.shape}: the same window and dimension expected")
    kernel = np.empty((len(A), len(B)))
    start, lam2, tau2 = 1 / (1 - lam ** 2), lam ** 2, tau ** 2
    for i in range(0, len(A), BLOCK):
        block, columns = A[i:i + BLOCK], B[i:] if symmetric else B
        K = np.full((len(block), len(columns)), start)
        for u in range(A.shape[1] - 1, -1, -1):  # (from the oldest lag to the current frame)
            G = block[:, u] @ columns[:, u].T
            G *= -tau2
            G += 1.0  # (1 - tau^2 <z_-u, z'_-u>, positive under the conditions of parameters)
            np.divide(K, G, out=K)
            K *= lam2
            K += 1.0
        if symmetric:
            kernel[i:i + BLOCK, i:] = K
            kernel[i + BLOCK:, i:i + BLOCK] = K[:, BLOCK:].T
        else:
            kernel[i:i + BLOCK] = K
    return kernel
