"""Step 4 - disclosure risk (Section 4.5): distance-to-closest-record statistics in
the standardised feature space against the leave-one-out real-to-real baseline.
Written over all variables and over the informative ones (`dcr_informative.csv`)."""
import pandas as pd

from _common import config, parser

from sdlm import data, metrics

if __name__ == "__main__":
    args = parser(__doc__).parse_args()
    out = config.out_dir(args.dataset) / "tables"
    real = data.load(args.dataset)

    for cols, tag in ((list(real.columns), ""), (metrics.informative_columns(real), "_informative")):
        rows = [{"dataset": "real-to-real (baseline)", **metrics.dcr_baseline(real[cols])}]
        for model in args.models:
            synth = data.load_synthetic(args.dataset, model)
            rows.append({"dataset": model, **metrics.dcr(real[cols], synth[cols])})
        res = pd.DataFrame(rows)
        res.to_csv(out / f"dcr{tag}.csv", index=False, float_format="%.4f")
        print(f"--- {'informative variables' if tag else 'all variables'} ({len(cols)})")
        print(res.round(3).to_string(index=False))
