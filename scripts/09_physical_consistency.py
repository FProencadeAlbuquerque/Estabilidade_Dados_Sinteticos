"""Step 9 - physical validity of the synthetic records (power-flow consistency).

For every synthetic record the AC power flow is solved for its injections and compared
with the dependent variables proposed by the generator (see `sdlm/physics.py`). The
score rho is calibrated on real records, for which it is practically zero. Records with
rho above a threshold tau (or whose power flow diverges) are rejected.

Outputs (results/<dataset>/):
    tables/physical_rho.csv            rho of every synthetic record, one column per model
    tables/physical_consistency.csv    convergence, rho quantiles, acceptance rate per tau
    tables/physical_tstr.csv           TSTR of: all records, accepted records, projected records
    figures/physical_rho.pdf           distribution of rho per generator
"""
import numpy as np
import pandas as pd

from _common import config, parser

from sdlm import data, metrics, physics, plotting

TAUS = (0.25, 0.5, 1.0)

if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--n-real", type=int, default=2000, help="real records used for calibration")
    p.add_argument("--jobs", type=int, default=2)
    args = p.parse_args()
    out = config.out_dir(args.dataset)
    real = data.load(args.dataset)
    train, test = data.split(real)

    cal_set = test.head(args.n_real)
    pf_real = physics.solve_table(args.dataset, cal_set, args.jobs)
    cal = physics.calibrate(real, cal_set, pf_real)
    rho_real = physics.inconsistency(cal_set, pf_real, cal)
    quasi = [c for c in pf_real.columns if c not in cal["dep"]]
    print(f"dependent variables scored: {cal['dep']}")
    print(f"real records: rho median {rho_real.median():.4f}, max {rho_real.max():.4f}")

    summary = [{"data": "real", "converged_pct": 100 * np.isfinite(rho_real).mean(),
                "rho_median": rho_real.median(), "rho_p90": rho_real.quantile(0.9), "rho_max": rho_real.max(),
                **{f"accepted_pct_tau{t}": 100 * (rho_real <= t).mean() for t in TAUS}}]
    tstr = [{"training_data": "real (TRTR)", "n": len(train), **metrics.tstr(train, test)}]
    rhos = {"real": rho_real.reset_index(drop=True)}
    for model in args.models:
        synth = data.load_synthetic(args.dataset, model)
        pf = physics.solve_table(args.dataset, synth, args.jobs)
        rho = physics.inconsistency(synth, pf, cal)
        rhos[model] = rho
        fin = rho[np.isfinite(rho)]
        row = {"data": model, "converged_pct": 100 * np.isfinite(rho).mean(), "rho_median": fin.median(),
               "rho_p90": fin.quantile(0.9), "rho_max": fin.max(),
               **{f"accepted_pct_tau{t}": 100 * (rho <= t).mean() for t in TAUS}}
        if quasi:      # set-point-bound voltages: absolute error in p.u.
            row["quasi_const_max_abs_err"] = (synth[quasi] - pf[quasi] - cal["bias"][quasi]).abs().max().max()
        summary.append(row)
        tstr.append({"training_data": f"{model}: all records", "n": len(synth), **metrics.tstr(synth, test)})
        for t in TAUS:
            acc = synth[rho <= t]
            if len(acc) >= 200:
                tstr.append({"training_data": f"{model}: rho <= {t}", "n": len(acc), **metrics.tstr(acc, test)})
        proj = physics.project(synth, pf, cal)
        tstr.append({"training_data": f"{model}: projected", "n": len(proj), **metrics.tstr(proj, test)})
        print(f"[{args.dataset}] {model} done", flush=True)

    tab = out / "tables"
    pd.DataFrame(rhos).to_csv(tab / "physical_rho.csv", index=False, float_format="%.5f")
    summary = pd.DataFrame(summary)
    summary.to_csv(tab / "physical_consistency.csv", index=False, float_format="%.4f")
    tstr = pd.DataFrame(tstr).drop(columns=["target_range"])
    tstr.to_csv(tab / "physical_tstr.csv", index=False, float_format="%.4f")
    plotting.rho_distribution({m: rhos[m] for m in args.models}, TAUS, out / "figures" / "physical_rho.pdf")
    print(summary.round(3).to_string(index=False))
    print(tstr.round(3).to_string(index=False))
