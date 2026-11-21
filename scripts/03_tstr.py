"""Step 3 - downstream utility (Section 4.4): train-on-real (TRTR) baseline versus
train-on-synthetic, test-on-real (TSTR) with a random-forest regressor."""
import pandas as pd

from _common import config, parser

from sdlm import data, metrics

if __name__ == "__main__":
    args = parser(__doc__).parse_args()
    out = config.out_dir(args.dataset) / "tables"
    train, test = data.split(data.load(args.dataset))

    rows = [{"training_data": "real (TRTR)", **metrics.tstr(train, test)}]
    for model in args.models:
        rows.append({"training_data": f"{model} (TSTR)",
                     **metrics.tstr(data.load_synthetic(args.dataset, model), test)})
    res = pd.DataFrame(rows)
    res.to_csv(out / "tstr.csv", index=False, float_format="%.4f")
    print(res.round(3).to_string(index=False))
