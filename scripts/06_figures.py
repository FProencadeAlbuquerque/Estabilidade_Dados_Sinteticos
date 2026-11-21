"""Step 6 - figures of the paper: CTGAN loss history (Fig. 5), KDE comparison (Fig. 6),
CDFs of the two features with the largest KS (Fig. 7) and UMAP projection (Fig. 8)."""
import pandas as pd

from _common import config, parser

from sdlm import data, plotting

if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--kde-features", nargs="+", default=["V9", "Theta2", "Q4", "LoadMargin"])
    args = p.parse_args()
    out = config.out_dir(args.dataset)
    real = data.load(args.dataset)
    synth = data.load_synthetic(args.dataset, "ctgan")
    fig = out / "figures"

    plotting.loss_history(pd.read_csv(out / "tables" / "ctgan_loss.csv"), fig / "ctgan_loss_history.pdf")
    plotting.kde_comparison(real, synth, args.kde_features, fig / "KDE_comparison.pdf")
    fid = pd.read_csv(out / "tables" / "fidelity_ctgan.csv", index_col=0)
    worst = fid.KS.sort_values(ascending=False).index[:2].tolist()
    plotting.cdf_comparison(real, synth, worst, fig / "CDF_comparison.pdf")
    plotting.umap_projection(real, synth, fig / "umap_manifold.pdf")
    print("figures written to", fig, "| CDF features:", worst)
