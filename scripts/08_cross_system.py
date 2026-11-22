"""Step 8 - cross-system summary (scalability analysis): gathers, for every dataset
already processed by steps 1-4, the size of the problem, the fidelity summary, the
TSTR errors and the DCR statistics of each generator in a single table. Fidelity and
DCR are taken over the informative variables (see `config.CV_MIN`) so that systems
with different numbers of set-point-bound voltage channels are comparable.

Outputs (results/):
    cross_system.csv     one row per (dataset, model)
    cross_system.tex     LaTeX rows ready to paste into the manuscript
"""
import numpy as np
import pandas as pd

from _common import config, parser

from sdlm import data, metrics

LABELS = {"ieee14": "IEEE 14 (bus 14)", "ieee14_n1": "IEEE 14, line 2-3 out (bus 14)",
          "ieee118_bus17": "IEEE 118 (bus 17)", "ieee118_bus45": "IEEE 118 (bus 45)"}

if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--datasets", nargs="+", default=["ieee14", "ieee118_bus45", "ieee118_bus17"])
    args = p.parse_args()

    rows = []
    for ds in args.datasets:
        tab = config.RESULTS / ds / "tables"
        real = data.load(ds)
        n_info = len(metrics.informative_columns(real))
        cmp_ = pd.read_csv(tab / "model_comparison_informative.csv").set_index("model")
        tstr = pd.read_csv(tab / "tstr.csv").set_index("training_data")
        dcr = pd.read_csv(tab / "dcr_informative.csv").set_index("dataset")
        base = dict(dataset=ds, n_columns=real.shape[1], n_informative=n_info)
        mdnn_base = pd.read_csv(tab / "mdnn_informative.csv")["real_to_real_loo"].iloc[0]
        rows.append({**base, "model": "real", "MDNN": mdnn_base, "NRMSE_pct": tstr.loc["real (TRTR)", "NRMSE_pct"],
                     "NMAE_pct": tstr.loc["real (TRTR)", "NMAE_pct"],
                     "DCR_mean": dcr.loc["real-to-real (baseline)", "mean"],
                     "DCR_p5": dcr.loc["real-to-real (baseline)", "p5"],
                     "DCR_min": dcr.loc["real-to-real (baseline)", "min"]})
        for m in args.models:
            if m not in cmp_.index:
                continue
            rows.append({**base, "model": m, **cmp_.loc[m].to_dict(),
                         "NRMSE_pct": tstr.loc[f"{m} (TSTR)", "NRMSE_pct"],
                         "NMAE_pct": tstr.loc[f"{m} (TSTR)", "NMAE_pct"],
                         "DCR_mean": dcr.loc[m, "mean"], "DCR_p5": dcr.loc[m, "p5"],
                         "DCR_min": dcr.loc[m, "min"]})
    res = pd.DataFrame(rows)
    res.to_csv(config.RESULTS / "cross_system.csv", index=False, float_format="%.4f")

    def f(v, nd=3):
        return "--" if pd.isna(v) else (r"$\infty$" if np.isinf(v) else f"{v:.{nd}f}")

    lines = []
    for ds, g in res.groupby("dataset", sort=False):
        lines.append(rf"\multicolumn{{9}}{{l}}{{\textit{{{LABELS.get(ds, ds)}, "
                     rf"{g.n_columns.iloc[0]} variables, {g.n_informative.iloc[0]} informative}}}} \\")
        for r in g.itertuples():
            name = "Real data" if r.model == "real" else config.MODEL_LABELS[r.model]
            lines.append(" & ".join([name, f(r.mean_HD), f(r.max_HD), f(r.mean_KS), f(r.max_KS),
                                     f(r.MDNN), f(r.NRMSE_pct, 1), f(r.DCR_p5), f(r.DCR_min)]) + r" \\")
        lines.append(r"\hline")
    (config.RESULTS / "cross_system.tex").write_text("\n".join(lines) + "\n")
    print(res.round(3).to_string(index=False))
