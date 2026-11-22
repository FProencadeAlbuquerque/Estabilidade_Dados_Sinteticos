"""Step 10 - robustness to a change of topology (Reviewer 2, comment 1).

Datasets: `ieee14` (intact network) and `ieee14_n1` (line 2-3 out of service), both
generated with the same sampling of the loads. The script measures

  (a) the load-margin error on the real N-1 test partition of regressors trained on
      intact-topology data (real or synthetic) versus N-1 data (real or synthetic);
  (b) whether the physical consistency check, run with the N-1 network, flags records
      that come from the intact topology (real or synthetic).

Requires steps 1 and 9 for both datasets. Output: results/cross_topology.csv
"""
import numpy as np
import pandas as pd

from _common import config, parser

from sdlm import data, metrics, physics

SRC, DST = "ieee14", "ieee14_n1"

if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--n-check", type=int, default=2000)
    p.add_argument("--jobs", type=int, default=2)
    args = p.parse_args()

    real_src, real_dst = data.load(SRC), data.load(DST)
    tr_src, _ = data.split(real_src)
    tr_dst, te_dst = data.split(real_dst)
    cal = physics.calibrate(real_dst, physics.split_columns(DST, real_dst.columns)[1] + [config.TARGET])

    def accepted_pct(table):
        t = table.head(args.n_check)
        r = physics.inconsistency(t, physics.solve_table(DST, t, args.jobs), cal)
        return 100 * (r.rho <= physics.TAU).mean()

    rows = []

    def add(train_on, table, check=True):
        t = metrics.tstr(table, te_dst)
        rows.append({"training_data": train_on, "n": len(table), "NRMSE_pct": t["NRMSE_pct"],
                     "NMAE_pct": t["NMAE_pct"], "accepted_by_N1_check_pct": accepted_pct(table) if check else np.nan})
        print(rows[-1], flush=True)

    add("intact topology: real", tr_src)
    for m in args.models:
        add(f"intact topology: {m}", data.load_synthetic(SRC, m))
    add("N-1 topology: real", tr_dst)
    for m in args.models:
        add(f"N-1 topology: {m}", data.load_synthetic(DST, m), check=False)
        add(f"N-1 topology: {m}, accepted", data.load_synthetic(DST, f"{m}_accepted"), check=False)
    res = pd.DataFrame(rows)
    res.to_csv(config.RESULTS / "cross_topology.csv", index=False, float_format="%.3f")
    print(res.round(2).to_string(index=False))
