"""Missing-value handling (Section 4.1): local-mean vs. generative completion."""
import numpy as np


def mcar_mask(n: int, rate: float, rng: np.random.Generator) -> np.ndarray:
    """Boolean mask with exactly round(rate * n) missing entries, chosen at random."""
    mask = np.zeros(n, dtype=bool)
    mask[rng.choice(n, size=int(round(rate * n)), replace=False)] = True
    return mask


def local_mean_impute(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """x(n) <- [x(n+k) + x(n-j)] / 2 with the nearest available neighbours on each
    side (k, j the smallest positive shifts with observed samples). At the borders,
    where only one side exists, that neighbour is used."""
    x = np.asarray(x, float)
    n = len(x)
    idx = np.arange(n)
    obs = ~mask
    prev = np.where(obs, idx, -1)
    np.maximum.accumulate(prev, out=prev)
    nxt = np.where(obs, idx, n)
    nxt = np.minimum.accumulate(nxt[::-1])[::-1]
    out = x.copy()
    for i in np.flatnonzero(mask):
        p, q = prev[i], nxt[i]
        vals = [x[j] for j in (p, q) if 0 <= j < n]
        out[i] = np.mean(vals)
    return out


def generative_impute(x: np.ndarray, mask: np.ndarray, x_synth: np.ndarray) -> np.ndarray:
    """Replace each missing entry by the value of the same feature in the corresponding
    record drawn from the trained generator (unconditional sampling, so the records are
    exchangeable); observed entries are retained."""
    out = np.asarray(x, float).copy()
    out[mask] = np.asarray(x_synth, float)[: len(out)][mask]
    return out
