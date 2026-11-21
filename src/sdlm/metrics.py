"""Fidelity (HD, KL, KS, MDNN), utility (TSTR) and privacy (DCR) metrics.

Definitions follow Sections 3.1-3.3 and Appendix B of the paper.
"""
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, ks_2samp
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors

from . import config

GRID = 1000
EPS = 1e-12


# --------------------------------------------------------------------------- marginals
def _is_const(x):
    return np.ptp(x) == 0


def kde_pair(xe, xs, m: int = GRID):
    """Scott-rule Gaussian KDEs of both samples on a common uniform mesh,
    renormalised so that each integrates to one on the mesh (Appendix B)."""
    he = np.std(xe, ddof=1) * len(xe) ** (-1 / 5)
    hs = np.std(xs, ddof=1) * len(xs) ** (-1 / 5)
    pad = 3 * max(he, hs)
    grid = np.linspace(min(xe.min(), xs.min()) - pad, max(xe.max(), xs.max()) + pad, m)
    dx = grid[1] - grid[0]
    pe = gaussian_kde(xe, bw_method="scott")(grid)
    ps = gaussian_kde(xs, bw_method="scott")(grid)
    return grid, pe / (pe.sum() * dx), ps / (ps.sum() * dx), dx


def hellinger(pe, ps, dx):
    return float(np.sqrt(0.5 * np.sum((np.sqrt(pe) - np.sqrt(ps)) ** 2) * dx))


def kl_div(pe, ps, dx):
    pe_, ps_ = np.maximum(pe, EPS), np.maximum(ps, EPS)
    return float(np.sum(pe_ * np.log(pe_ / ps_)) * dx)


def marginal_metrics(xe, xs):
    """(HD, KL, KS) for one feature. Constant features with the same value -> zeros."""
    xe, xs = np.asarray(xe, float), np.asarray(xs, float)
    if _is_const(xe) and _is_const(xs):
        same = np.isclose(xe[0], xs[0])
        return (0.0, 0.0, 0.0) if same else (1.0, np.inf, 1.0)
    ks = float(ks_2samp(xe, xs).statistic)
    if _is_const(xe) or _is_const(xs):
        return 1.0, np.inf, ks
    _, pe, ps, dx = kde_pair(xe, xs)
    return hellinger(pe, ps, dx), kl_div(pe, ps, dx), ks


def fidelity_table(real: pd.DataFrame, synth: pd.DataFrame) -> pd.DataFrame:
    rows = [(c, *marginal_metrics(real[c].values, synth[c].values)) for c in real.columns]
    return pd.DataFrame(rows, columns=["feature", "HD", "KL", "KS"]).set_index("feature")


def informative_columns(real: pd.DataFrame, cv_min: float = config.CV_MIN, target: str = config.TARGET):
    """Columns whose coefficient of variation is at least `cv_min` (plus the target)."""
    sd, mu = real.std(ddof=0), real.mean().abs()
    cv = (sd / mu.where(mu > 0, np.nan)).fillna(0.0)
    return [c for c in real.columns if c == target or cv[c] >= cv_min]


# --------------------------------------------------------------------------- geometry
def standardise(real: pd.DataFrame, *others):
    """z-score every table with the mean/std of the empirical dataset."""
    mu = real.values.mean(0)
    sd = real.values.std(0)
    sd[sd == 0] = 1.0
    return [(t[real.columns].values - mu) / sd for t in (real, *others)]


def nn_dist(a, b):
    """Distance from each row of `a` to its nearest neighbour in `b`."""
    return NearestNeighbors(n_neighbors=1).fit(b).kneighbors(a)[0][:, 0]


def loo_dist(a):
    """Leave-one-out nearest-neighbour distance inside `a`."""
    return NearestNeighbors(n_neighbors=2).fit(a).kneighbors(a)[0][:, 1]


def mdnn(real: pd.DataFrame, synth: pd.DataFrame) -> dict:
    zr, zs = standardise(real, synth)
    r2s, s2r = nn_dist(zr, zs).mean(), nn_dist(zs, zr).mean()
    return {"real_to_synth": r2s, "synth_to_real": s2r, "MDNN": 0.5 * (r2s + s2r),
            "real_to_real_loo": loo_dist(zr).mean()}


# --------------------------------------------------------------------------- privacy
def _dcr_stats(d, exact_tol=1e-8):
    return {"mean": d.mean(), "p5": np.percentile(d, 5), "min": d.min(),
            "exact_copies": int((d < exact_tol).sum())}


def dcr(real: pd.DataFrame, synth: pd.DataFrame) -> dict:
    zr, zs = standardise(real, synth)
    return _dcr_stats(nn_dist(zs, zr))


def dcr_baseline(real: pd.DataFrame) -> dict:
    (zr,) = standardise(real)
    out = _dcr_stats(loo_dist(zr))
    out["exact_copies"] = np.nan
    return out


# --------------------------------------------------------------------------- utility
def tstr(train: pd.DataFrame, test: pd.DataFrame, target: str = config.TARGET) -> dict:
    """Train a random forest on `train`, evaluate on the real `test` partition.
    Errors are normalised by the range of the target in the test set."""
    feats = [c for c in test.columns if c != target]
    rf = RandomForestRegressor(**config.RF_PARAMS).fit(train[feats], train[target])
    err = rf.predict(test[feats]) - test[target].values
    rng = test[target].max() - test[target].min()
    rmse, mae = np.sqrt(np.mean(err ** 2)), np.mean(np.abs(err))
    return {"RMSE": rmse, "MAE": mae, "NRMSE_pct": 100 * rmse / rng, "NMAE_pct": 100 * mae / rng,
            "target_range": rng}
