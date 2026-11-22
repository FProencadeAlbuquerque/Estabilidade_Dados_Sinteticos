"""Step 9 - physical validity of the synthetic records.

For every synthetic record the AC power flow and the nose point are computed for its
injections and compared with the states and the load margin proposed by the generator
(see `sdlm/physics.py`). Records with rho > tau (default 1), or whose power flow does
not converge, are rejected.

Outputs (results/<dataset>/):
    tables/physical_rho_<model>.csv    rho_state, rho_lm, rho of every synthetic record
    tables/physical_consistency.csv    convergence, rho quantiles, acceptance rates
    tables/physical_filter.csv         TSTR, fidelity and DCR of: all records, accepted
                                       records and physics-projected records
    synthetic/<model>_accepted.csv     records that pass the filter
    synthetic/<model>_projected.csv    generated injections + network solution
    figures/physical_rho.pdf           distribution of rho per generator
"""
import numpy as np
import pandas as pd

from _common import config, parser

from sdlm import data, metrics, physics, plotting


def evaluate(name, table, real, test, info):
    fid = metrics.fidelity_table(real[info], table[info])
    d = metrics.dcr(real[info], table[info])
    t = metrics.tstr(table, test)
    return {"training_data": name, "n": len(table), "NRMSE_pct": t["NRMSE_pct"], "NMAE_pct": t["NMAE_pct"],
            "mean_HD": fid.HD.mean(), "max_HD": fid.HD.max(), "DCR_p5": d["p5"], "DCR_min": d["min"]}


if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--tau", type=float, default=physics.TAU)
    p.add_argument("--n-real", type=int, default=2000, help="real records used to validate the network model")
    p.add_argument("--jobs", type=int, default=2)
    args = p.parse_args()
    out = config.out_dir(args.dataset)
    tab = out / "tables"
    real = data.load(args.dataset)
    train, test = data.split(real)
    info = metrics.informative_columns(real)

    chk = test.head(args.n_real)
    pf_real = physics.solve_table(args.dataset, chk, args.jobs)
    cal = physics.calibrate(real, pf_real.columns)
    r_real = physics.inconsistency(chk, pf_real, cal)
    quasi = [c for c in pf_real.columns if c not in cal["states"] and c != config.TARGET]
    print(f"states scored: {cal['states']}")
    print(f"real records: rho_state max {r_real.rho_state.max():.4f}, rho_lm max {r_real.rho_lm.max():.4f}")

    def stats(name, r):
        fin = r[np.isfinite(r.rho)]
        return {"data": name, "converged_pct": 100 * np.isfinite(r.rho).mean(),
                "rho_state_median": fin.rho_state.median(), "rho_state_p90": fin.rho_state.quantile(0.9),
                "rho_lm_median": fin.rho_lm.median(), "rho_lm_p90": fin.rho_lm.quantile(0.9),
                "accepted_state_pct": 100 * (r.rho_state <= args.tau).mean(),
                "accepted_lm_pct": 100 * (r.rho_lm <= args.tau).mean(),
                "accepted_pct": 100 * (r.rho <= args.tau).mean()}

    summary = [stats("real", r_real)]
    rows = [{"training_data": "real (TRTR)", "n": len(train), **{k: v for k, v in metrics.tstr(train, test).items()
                                                               if k in ("NRMSE_pct", "NMAE_pct")},
             "DCR_p5": metrics.dcr_baseline(real[info])["p5"], "DCR_min": metrics.dcr_baseline(real[info])["min"]}]
    rhos = {}
    for model in args.models:
        synth = data.load_synthetic(args.dataset, model)
        pf = physics.solve_table(args.dataset, synth, args.jobs)
        r = physics.inconsistency(synth, pf, cal)
        r.to_csv(tab / f"physical_rho_{model}.csv", index=False, float_format="%.5f")
        rhos[model] = r.rho
        row = stats(model, r)
        if quasi:      # set-point-bound voltages: absolute error in p.u.
            row["quasi_const_max_abs_err"] = (synth[quasi] - pf[quasi]).abs().max().max()
        summary.append(row)
        accepted = synth[r.rho <= args.tau]
        projected = physics.project(synth, pf)
        accepted.to_csv(out / "synthetic" / f"{model}_accepted.csv", index=False)
        projected.to_csv(out / "synthetic" / f"{model}_projected.csv", index=False)
        rows.append(evaluate(f"{model}: all records", synth, real, test, info))
        if len(accepted) >= 200:
            rows.append(evaluate(f"{model}: accepted (rho <= {args.tau:g})", accepted, real, test, info))
        rows.append(evaluate(f"{model}: projected", projected, real, test, info))
        print(f"[{args.dataset}] {model} done", flush=True)

    summary = pd.DataFrame(summary)
    summary.to_csv(tab / "physical_consistency.csv", index=False, float_format="%.4f")
    res = pd.DataFrame(rows)
    res.to_csv(tab / "physical_filter.csv", index=False, float_format="%.4f")
    plotting.rho_distribution(rhos, (args.tau,), out / "figures" / "physical_rho.pdf")
    pd.set_option("display.width", 250)
    print(summary.round(3).to_string(index=False))
    print(res.round(3).to_string(index=False))
