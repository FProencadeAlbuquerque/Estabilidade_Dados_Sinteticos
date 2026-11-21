"""Step 1 - fit CTGAN / TVAE / Gaussian copula on the 8,000-sample training partition
and draw 10,000 synthetic records from each (Sections 4 and 4.3).

Outputs (results/<dataset>/):
    synthetic/<model>.csv        synthetic records
    tables/ctgan_loss.csv        per-epoch generator / discriminator losses
    models/<model>.pkl           fitted synthesizer (not versioned)
"""
import time

from _common import config, parser

from sdlm import data, generators

if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--epochs", type=int, default=None, help="override the number of epochs (quick tests)")
    args = p.parse_args()

    out = config.out_dir(args.dataset)
    train, _ = data.split(data.load(args.dataset))
    for model in args.models:
        t0 = time.time()
        synth, syn = generators.fit_sample(model, train, epochs=args.epochs)
        synth.to_csv(out / "synthetic" / f"{model}.csv", index=False)
        syn.save(out / "models" / f"{model}.pkl")
        if model == "ctgan":
            syn.get_loss_values().to_csv(out / "tables" / "ctgan_loss.csv", index=False)
        print(f"[{args.dataset}] {model}: {len(synth)} rows, {time.time() - t0:.0f} s", flush=True)
