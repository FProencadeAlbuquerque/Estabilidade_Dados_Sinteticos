"""Figures of the paper (matplotlib, vector PDF with embedded Type-42 fonts)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from . import config, metrics

plt.rcParams.update({
    "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 13, "legend.fontsize": 11,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True, "savefig.bbox": "tight",
    "mathtext.fontset": "dejavusans",
})
C_REAL, C_SYN, C_ALT = "#1f77b4", "#ff7f0e", "#2ca02c"


def label(col: str) -> str:
    """'Theta9' -> r'$\\theta_{9}$', 'P14' -> '$P_{14}$'."""
    if col == config.TARGET:
        return "Load margin (MW)"
    for prefix, sym in (("Theta", r"\theta"), ("V", "V"), ("P", "P"), ("Q", "Q")):
        if col.startswith(prefix) and col[len(prefix):].isdigit():
            return rf"${sym}_{{{col[len(prefix):]}}}$"
    return col


def _kde(x, grid):
    return gaussian_kde(x, bw_method="scott")(grid)


def _grid(*xs, m=500):
    lo, hi = min(x.min() for x in xs), max(x.max() for x in xs)
    pad = 0.05 * (hi - lo)
    return np.linspace(lo - pad, hi + pad, m)


def _save(fig, path):
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=150)
    plt.close(fig)


def missing_kde(x, completed: dict, feat: str, path, model_label="CTGAN"):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, (rate, (lm, gan)) in zip(axes.ravel(), completed.items()):
        g = _grid(x, lm, gan)
        ax.plot(g, _kde(x, g), color=C_REAL, lw=2.2, label="Original")
        ax.plot(g, _kde(lm, g), color=C_SYN, lw=2, ls="--", label="Local mean")
        ax.plot(g, _kde(gan, g), color=C_ALT, lw=1.6, ls="-.", label=model_label)
        ax.set_title(f"{int(100 * rate)}% missing")
        ax.set_xlabel(label(feat))
        ax.set_ylabel("Density")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=len(h), frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, path)


def kde_comparison(real, synth, feats, path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, f in zip(axes.ravel(), feats):
        g = _grid(real[f].values, synth[f].values)
        ax.plot(g, _kde(real[f].values, g), color=C_REAL, lw=2.2, label="Empirical")
        ax.plot(g, _kde(synth[f].values, g), color=C_SYN, lw=2, ls="--", label="Synthetic (CTGAN)")
        ax.set_xlabel(label(f))
        ax.set_ylabel("Density")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=len(h), frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, path)


def cdf_comparison(real, synth, feats, path):
    fig, axes = plt.subplots(1, len(feats), figsize=(5 * len(feats), 4))
    for ax, f in zip(np.atleast_1d(axes), feats):
        for x, c, ls, lab in ((real[f].values, C_REAL, "-", "Empirical"),
                              (synth[f].values, C_SYN, "--", "Synthetic (CTGAN)")):
            xs = np.sort(x)
            ax.plot(xs, np.arange(1, len(xs) + 1) / len(xs), color=c, ls=ls, lw=2, label=lab)
        ax.set_xlabel(label(f))
        ax.set_ylabel("CDF")
    np.atleast_1d(axes)[0].legend(frameon=True)
    fig.tight_layout()
    _save(fig, path)


def loss_history(loss, path, alpha=0.02):
    ep = loss["Epoch"].values
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for col, ls, lab in (("Generator Loss", "-", "Generator"), ("Discriminator Loss", "--", "Discriminator")):
        y = loss[col].values
        ax.plot(ep, y, color="0.75", lw=0.6, label="Per-epoch loss" if ls == "-" else None)
        ax.plot(ep, loss[col].ewm(alpha=alpha).mean().values, color="k" if ls == "-" else "0.3",
                lw=2, ls=ls, label=lab)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_xlim(0, ep.max() + 1)
    ax.legend(frameon=True)
    _save(fig, path)


def umap_projection(real, synth, path, n=3000, seed=config.SEED, model_label="CTGAN"):
    import umap

    zr, zs = metrics.standardise(real, synth)
    rng = np.random.default_rng(seed)
    zr = zr[rng.choice(len(zr), min(n, len(zr)), replace=False)]
    zs = zs[rng.choice(len(zs), min(n, len(zs)), replace=False)]
    emb = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=seed).fit_transform(np.vstack([zr, zs]))
    is_syn = np.r_[np.zeros(len(zr), bool), np.ones(len(zs), bool)]
    order = rng.permutation(len(emb))          # interleave so neither set hides the other
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(*emb[order].T, s=7, c=np.where(is_syn[order], C_SYN, C_REAL), alpha=0.5, lw=0)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    handles = [plt.Line2D([], [], ls="", marker="o", ms=7, color=c, label=lab)
               for c, lab in ((C_REAL, "Empirical"), (C_SYN, f"Synthetic ({model_label})"))]
    ax.legend(handles=handles, frameon=True)
    _save(fig, path)
