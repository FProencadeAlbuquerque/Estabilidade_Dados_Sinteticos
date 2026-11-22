"""Physical consistency of a tabular sample with the AC power-flow equations.

Every record of the datasets is a solved operating point: the injections (loads and
generator active powers) are the independent variables, and the PMU phasors, the slack
power and the generator reactive powers are *dependent* on them through the network.
For a synthetic record (u_s, y_s) the power flow is solved for the injections u_s and
the result is compared with the dependent variables y_s proposed by the generator:

    e_j   = y_s,j - PF_j(u_s) - b_j          (b_j: model offset, estimated on real data)
    rho   = sqrt( mean_j (e_j / sigma_j)^2 ) (sigma_j: std of variable j in the real data)

rho = 0 means that the record satisfies the power-flow equations; rho = 1 means that
its dependent variables are as far from the solution as a randomly chosen operating
point. Records whose power flow does not converge are infeasible (rho = inf).

Networks: MATPOWER case14 / case118 as shipped with pandapower. Injections that are
not columns of a dataset stay at their base-case values.
"""
import warnings

import numpy as np
import pandas as pd

from . import config, metrics

warnings.filterwarnings("ignore")

PMU_118 = [2, 3, 5, 11, 13, 18, 19, 29, 32, 56, 57, 80, 81, 88, 95]

# network, columns that are generator active powers / reactive outputs, slack column
SYSTEMS = {
    "ieee14": dict(case="case14", gen_p=(), gen_q=(), slack_p="P1", extra_load_buses=(7,)),
    "ieee14_contingency": dict(case="case14", gen_p=(), gen_q=(), slack_p="P1", extra_load_buses=(7,)),
    "ieee118_bus17": dict(case="case118", gen_p=(46, 49), gen_q=(49,), slack_p=None, extra_load_buses=()),
    "ieee118_bus45": dict(case="case118", gen_p=(46, 49), gen_q=(49,), slack_p=None, extra_load_buses=()),
}


def _bus(col):
    for prefix in ("Theta", "V", "P", "Q"):
        if col.startswith(prefix) and col[len(prefix):].isdigit():
            return prefix, int(col[len(prefix):])
    return None, None


def split_columns(dataset: str, columns):
    """(independent injections, dependent variables) among the columns of a dataset."""
    s = SYSTEMS[dataset]
    indep, dep = [], []
    for c in columns:
        kind, b = _bus(c)
        if c == config.TARGET or kind is None:
            continue
        if kind in ("V", "Theta") or c == s["slack_p"] or (kind == "Q" and b in s["gen_q"]):
            dep.append(c)
        else:
            indep.append(c)
    return indep, dep


class PowerFlow:
    """Solves the power flow for the injections of a record and returns its dependent variables."""

    def __init__(self, dataset: str, columns):
        import pandapower as pp
        import pandapower.networks as pn

        self.pp, self.s = pp, SYSTEMS[dataset]
        self.net = getattr(pn, self.s["case"])()
        for b in self.s["extra_load_buses"]:          # buses without load in the base case
            pp.create_load(self.net, b - 1, 0.0, 0.0)
        self.load = {int(b) + 1: i for i, b in zip(self.net.load.index, self.net.load.bus)}
        self.gen = {int(b) + 1: i for i, b in zip(self.net.gen.index, self.net.gen.bus)}
        self.indep, self.dep = split_columns(dataset, columns)
        pp.runpp(self.net)

    def solve(self, row) -> dict | None:
        net, s = self.net, self.s
        for c in self.indep:
            kind, b = _bus(c)
            if kind == "P" and b in s["gen_p"]:
                net.gen.at[self.gen[b], "p_mw"] = row[c]
            else:
                net.load.at[self.load[b], "p_mw" if kind == "P" else "q_mvar"] = row[c]
        try:
            self.pp.runpp(net, init="flat", max_iteration=30)
        except Exception:
            return None
        out = {}
        for c in self.dep:
            kind, b = _bus(c)
            if kind == "V":
                out[c] = net.res_bus.vm_pu.iat[b - 1]
            elif kind == "Theta":
                out[c] = net.res_bus.va_degree.iat[b - 1]
            elif c == s["slack_p"]:
                out[c] = net.res_ext_grid.p_mw.iat[0]
            else:
                out[c] = net.res_gen.q_mvar.at[self.gen[b]]
        return out


def _solve_chunk(args):
    dataset, df = args
    pf = PowerFlow(dataset, df.columns)
    rows = [pf.solve(r) or {c: np.nan for c in pf.dep} for r in df.to_dict("records")]
    return pd.DataFrame(rows, index=df.index)


def solve_table(dataset: str, df: pd.DataFrame, n_jobs: int = 2) -> pd.DataFrame:
    """Power-flow solution of the dependent variables for every record (NaN: no convergence)."""
    from multiprocessing import Pool

    chunks = [(dataset, c) for c in np.array_split(df, max(1, n_jobs * 4))]
    with Pool(n_jobs) as pool:
        return pd.concat(pool.map(_solve_chunk, chunks)).loc[df.index]


def calibrate(real: pd.DataFrame, cal_set: pd.DataFrame, pf_cal: pd.DataFrame) -> dict:
    """Model offset b_j (mean residual of the real records `cal_set`, whose power-flow
    solution is `pf_cal`) and scale sigma_j (std in the full real dataset). Only
    informative dependent variables enter the score; set-point-bound voltages are
    reported separately, in absolute value."""
    info = set(metrics.informative_columns(real))
    dep = [c for c in pf_cal.columns if c in info]
    bias = (cal_set[pf_cal.columns].values - pf_cal.values).mean(axis=0)
    return {"dep": dep, "bias": pd.Series(bias, index=pf_cal.columns), "sigma": real[dep].std(ddof=0)}


def inconsistency(table: pd.DataFrame, pf: pd.DataFrame, cal: dict) -> pd.Series:
    """Physical inconsistency score rho of each record (inf when the power flow diverges)."""
    e = (table[cal["dep"]] - pf[cal["dep"]] - cal["bias"][cal["dep"]]) / cal["sigma"]
    rho = np.sqrt((e ** 2).mean(axis=1))
    return rho.where(pf[cal["dep"]].notna().all(axis=1), np.inf).rename("rho")


def project(table: pd.DataFrame, pf: pd.DataFrame, cal: dict) -> pd.DataFrame:
    """Physics-corrected table: dependent variables replaced by the power-flow solution."""
    out = table.copy()
    ok = pf.notna().all(axis=1)
    out.loc[ok, pf.columns] = (pf + cal["bias"]).loc[ok, pf.columns].values
    return out[ok]
