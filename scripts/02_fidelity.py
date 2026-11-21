"""Step 2 - statistical fidelity (Sections 4.2 and 4.3): per-feature HD / KL / KS,
bidirectional MDNN and the model-comparison summary. The reference is the full
empirical dataset (N_s = 10,000).

Every summary is written twice: over all variables (as in the submitted manuscript) and
over the informative variables only (`*_informative.csv`, see `config.CV_MIN`)."""
import pandas as pd

from _common import config, parser

from sdlm import data, metrics

if __name__ == "__main__":
    args = parser(__doc__).parse_args()
    out = config.out_dir(args.dataset) / "tables"
    real = data.load(args.dataset)
    info = metrics.informative_columns(real)
    dropped = [c for c in real.columns if c not in info]
    print(f"{len(info)} informative variables of {real.shape[1]}; quasi-constant: {dropped}")
    pd.Series(dropped, name="quasi_constant").to_csv(out / "quasi_constant_columns.csv", index=False)

    for cols, tag in ((list(real.columns), ""), (info, "_informative")):
        summary, geo = [], []
        for model in args.models:
            synth = data.load_synthetic(args.dataset, model)
            fid = metrics.fidelity_table(real[cols], synth[cols])
            if not tag:
                fid.to_csv(out / f"fidelity_{model}.csv", float_format="%.4f")
            g = metrics.mdnn(real[cols], synth[cols])
            geo.append({"model": model, **g})
            summary.append({"model": model, "mean_HD": fid.HD.mean(), "max_HD": fid.HD.max(),
                            "mean_KL": fid.KL.mean(), "mean_KS": fid.KS.mean(), "max_KS": fid.KS.max(),
                            "MDNN": g["MDNN"]})
        pd.DataFrame(geo).to_csv(out / f"mdnn{tag}.csv", index=False, float_format="%.4f")
        summary = pd.DataFrame(summary)
        summary.to_csv(out / f"model_comparison{tag}.csv", index=False, float_format="%.4f")
        print(f"--- {'informative variables' if tag else 'all variables'} ({len(cols)})")
        print(summary.round(3).to_string(index=False))
        print(pd.DataFrame(geo).round(4).to_string(index=False))
