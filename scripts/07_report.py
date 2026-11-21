"""Step 7 - side-by-side comparison between the reproduced results and the values
reported in the manuscript (paper/reference_<dataset>.json). Writes
results/<dataset>/REPRODUCTION.md."""
import json

import pandas as pd

from _common import config, parser


def md(df: pd.DataFrame, fmt="%.3f") -> str:
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: "---" if pd.isna(v) else fmt % v)
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    return "\n".join([head, sep] + ["| " + " | ".join(map(str, r)) + " |" for r in df.values])


def side_by_side(repro: pd.DataFrame, ref: dict, key: str, cols) -> pd.DataFrame:
    rows = []
    for _, r in repro.iterrows():
        row = {key: r[key]}
        for c in cols:
            row[f"{c} (paper)"] = ref.get(r[key], {}).get(c, float("nan"))
            row[f"{c} (repro)"] = r[c]
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    args = parser(__doc__).parse_args()
    out = config.out_dir(args.dataset)
    tab = out / "tables"
    ref = json.loads((config.ROOT / "paper" / f"reference_{args.dataset}.json").read_text())

    parts = [f"# Reproduction report - `{args.dataset}`\n",
             "Reproduced values (this repository) next to the values reported in the manuscript. "
             "Real-data quantities (TRTR, real-to-real baseline) are deterministic and must match; "
             "generator-dependent quantities vary with library version, hardware and RNG stream.\n"]

    fid = pd.read_csv(tab / "fidelity_ctgan.csv")
    rows = [{"feature": r.feature, "HD (paper)": ref["fidelity_ctgan"][r.feature][0], "HD (repro)": r.HD,
             "KL (paper)": ref["fidelity_ctgan"][r.feature][1], "KL (repro)": r.KL,
             "KS (paper)": ref["fidelity_ctgan"][r.feature][2], "KS (repro)": r.KS} for r in fid.itertuples()]
    parts += ["## Per-feature fidelity, CTGAN (Table 1)\n", md(pd.DataFrame(rows), "%.4f"), ""]

    g = pd.read_csv(tab / "mdnn.csv").set_index("model").loc["ctgan"]
    rows = [{"indicator": k, "paper": v, "repro": g[k]} for k, v in ref["mdnn_ctgan"].items()]
    parts += ["## MDNN, CTGAN (Table 2)\n", md(pd.DataFrame(rows), "%.4f"), ""]

    cols = ["mean_HD", "max_HD", "mean_KL", "mean_KS", "max_KS", "MDNN"]
    parts += ["## Model comparison (Table 3)\n",
              md(side_by_side(pd.read_csv(tab / "model_comparison.csv"), ref["model_comparison"], "model", cols)), ""]
    parts += ["## TSTR (Table 4)\n",
              md(side_by_side(pd.read_csv(tab / "tstr.csv"), ref["tstr"], "training_data",
                              ["NRMSE_pct", "NMAE_pct"]), "%.2f"), ""]
    parts += ["## DCR (Table 5)\n",
              md(side_by_side(pd.read_csv(tab / "dcr.csv"), ref["dcr"], "dataset", ["mean", "p5", "min"])), ""]
    mv = tab / "missing_values.csv"
    if mv.exists():
        parts += ["## Missing-value completion (Figs. 3-4, quantified)\n",
                  "Distance between the completed feature and the original one "
                  "(`std_ratio` = std of completed / std of original).\n",
                  md(pd.read_csv(mv), "%.4f"), ""]
    (out / "REPRODUCTION.md").write_text("\n".join(parts))
    print("written", out / "REPRODUCTION.md")
