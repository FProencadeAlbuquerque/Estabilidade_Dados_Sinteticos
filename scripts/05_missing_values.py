"""Step 5 - handling missing values (Section 4.1): CTGAN completion versus local-mean
imputation at 5/10/20/30 % missing entries (MCAR), for the features of Figs. 3-4.

For each missing entry the value of the same feature in a CTGAN-generated record is
used; observed entries are kept. Besides the KDE figures, the script writes the
HD / KL / KS of each completed feature with respect to the original one."""
import numpy as np
import pandas as pd

from _common import config, parser

from sdlm import data, imputation, metrics, plotting

if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--features", nargs="+", default=None)
    p.add_argument("--model", default="ctgan")
    args = p.parse_args()
    out = config.out_dir(args.dataset)
    features = args.features or config.FEATURES[args.dataset]["missing"]
    real = data.load(args.dataset)
    synth = data.load_synthetic(args.dataset, args.model)

    rows = []
    for feat in features:
        x, completed = real[feat].values, {}
        for rate in config.MISSING_RATES:
            rng = np.random.default_rng(config.SEED)
            mask = imputation.mcar_mask(len(x), rate, rng)
            lm = imputation.local_mean_impute(x, mask)
            gan = imputation.generative_impute(x, mask, synth[feat].values)
            completed[rate] = (lm, gan)
            for name, xi in (("local-mean", lm), (args.model, gan)):
                hd, kl, ks = metrics.marginal_metrics(x, xi)
                rows.append({"feature": feat, "missing_pct": int(100 * rate), "method": name,
                             "HD": hd, "KL": kl, "KS": ks, "std_ratio": xi.std() / x.std()})
        plotting.missing_kde(x, completed, feat, out / "figures" / f"KDE_{feat}_missing.pdf",
                             model_label=config.MODEL_LABELS[args.model])
    res = pd.DataFrame(rows)
    res.to_csv(out / "tables" / "missing_values.csv", index=False, float_format="%.4f")
    print(res.round(4).to_string(index=False))
