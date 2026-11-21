"""Step 4 - disclosure risk (Section 4.5): distance-to-closest-record statistics in
the standardised feature space against the leave-one-out real-to-real baseline."""
import pandas as pd

from _common import config, parser

from sdlm import data, metrics

if __name__ == "__main__":
    args = parser(__doc__).parse_args()
    out = config.out_dir(args.dataset) / "tables"
    real = data.load(args.dataset)

    rows = [{"dataset": "real-to-real (baseline)", **metrics.dcr_baseline(real)}]
    for model in args.models:
        rows.append({"dataset": model, **metrics.dcr(real, data.load_synthetic(args.dataset, model))})
    res = pd.DataFrame(rows)
    res.to_csv(out / "dcr.csv", index=False, float_format="%.4f")
    print(res.round(3).to_string(index=False))
