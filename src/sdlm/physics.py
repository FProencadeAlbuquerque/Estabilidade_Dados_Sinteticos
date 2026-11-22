"""Physical consistency of a tabular record with the AC power-flow equations.

Every record of the datasets is a solved operating point: the injections (loads and
generator active powers) are the independent variables u; the PMU phasors, the slack
power and the generator reactive power (states y) and the load margin are functions
of u through the network. For a synthetic record the power flow and the nose point
are computed for its injections u_s and compared with what the generator proposed:

    rho_state = sqrt( mean_j ((y_s,j - PF_j(u_s)) / sigma_j)^2 )   informative states
    rho_lm    = |LM_s - LM(u_s)| / sigma_LM
    rho       = max(rho_state, rho_lm)

sigma is the standard deviation of the variable in the real data, so rho = 0 means a
record that satisfies the network equations and rho = 1 a record whose dependent
variables are as wrong as those of a randomly chosen operating point. Records with
rho > tau, or whose power flow does not converge, are rejected.

The networks and the load-margin definition reproduce the MATLAB/MATPOWER scripts
that generated the data (see `network.py`); real records give rho < 0.01.
"""
import numpy as np
import pandas as pd

from . import config, metrics
from .network import GROWTH, Network

TAU = 1.0

# network, target bus, columns that are generator active powers / reactive outputs
SYSTEMS = {
    "ieee14": dict(case="case14", target=14, gen_p=(), gen_q=(), slack_p="P1"),
    "ieee14_contingency": dict(case="case14", target=14, gen_p=(), gen_q=(), slack_p="P1"),
    "ieee118_bus17": dict(case="case118", target=17, gen_p=(46, 49), gen_q=(49,), slack_p=None),
    "ieee118_bus45": dict(case="case118", target=45, gen_p=(46, 49), gen_q=(49,), slack_p=None),
}


def _bus(col):
    for prefix in ("Theta", "V", "P", "Q"):
        if col.startswith(prefix) and col[len(prefix):].isdigit():
            return prefix, int(col[len(prefix):])
    return None, None


def split_columns(dataset: str, columns):
    """(independent injections, dependent states) among the columns of a dataset."""
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
    """Dependent variables (states and load margin) implied by the injections of a record."""

    def __init__(self, dataset: str, columns):
        self.s = SYSTEMS[dataset]
        self.net = Network(self.s["case"])
        self.indep, self.dep = split_columns(dataset, columns)
        self.target = self.net.ext[self.s["target"]]

    def solve(self, row) -> dict:
        net, s = self.net, self.s
        pd_, qd, pg = net.pd0.copy(), net.qd0.copy(), net.pg0.copy()
        for c in self.indep:
            kind, b = _bus(c)
            if kind == "P" and b in s["gen_p"]:
                pg[net.gidx[net.ext[b]]] = row[c]
            elif kind == "P":
                pd_[net.ext[b]] = row[c]
            else:
                qd[net.ext[b]] = row[c]
        V, ok, _ = net.solve(pd_, qd, pg)
        if not ok:
            return {c: np.nan for c in (*self.dep, config.TARGET)}
        out = {}
        for c in self.dep:
            kind, b = _bus(c)
            if kind == "V":
                out[c] = abs(V[net.ext[b]])
            elif kind == "Theta":
                out[c] = np.rad2deg(np.angle(V[net.ext[b]]))
            elif c == s["slack_p"]:
                out[c] = net.slack_p(V, pd_)
            else:
                out[c] = net.gen_q(V, qd, net.ext[b])
        # nose point with the base-case dispatch, as in the generating scripts
        lam = net.lambda_max(pd_, qd, net.pg0, V)
        out[config.TARGET] = GROWTH * lam * pd_[self.target]
        return out


def _solve_chunk(args):
    dataset, df = args
    pf = PowerFlow(dataset, df.columns)
    return pd.DataFrame([pf.solve(r) for r in df.to_dict("records")], index=df.index)


def solve_table(dataset: str, df: pd.DataFrame, n_jobs: int = 2) -> pd.DataFrame:
    """Network solution (states and load margin) for every record (NaN: no convergence)."""
    from multiprocessing import Pool

    chunks = [(dataset, c) for c in np.array_split(df, max(1, n_jobs * 8))]
    with Pool(n_jobs) as pool:
        return pd.concat(pool.map(_solve_chunk, chunks)).loc[df.index]


def calibrate(real: pd.DataFrame, pf_columns) -> dict:
    """Scales sigma_j from the real dataset. Only informative states enter rho_state;
    set-point-bound voltages are reported separately, in absolute value."""
    info = set(metrics.informative_columns(real))
    states = [c for c in pf_columns if c in info and c != config.TARGET]
    return {"states": states, "sigma": real[states].std(ddof=0), "sigma_lm": real[config.TARGET].std(ddof=0)}


def inconsistency(table: pd.DataFrame, pf: pd.DataFrame, cal: dict) -> pd.DataFrame:
    """rho_state, rho_lm and rho of each record (inf when the power flow diverges)."""
    st = cal["states"]
    e = (table[st] - pf[st]) / cal["sigma"]
    out = pd.DataFrame({"rho_state": np.sqrt((e ** 2).mean(axis=1)),
                        "rho_lm": (table[config.TARGET] - pf[config.TARGET]).abs() / cal["sigma_lm"]})
    out["rho"] = out.max(axis=1)
    return out.where(pf.notna().all(axis=1), np.inf)


def project(table: pd.DataFrame, pf: pd.DataFrame) -> pd.DataFrame:
    """Physics-corrected table: states and load margin replaced by the network solution,
    injections kept as generated."""
    ok = pf.notna().all(axis=1)
    out = table[ok].copy()
    out[pf.columns] = pf.loc[ok, pf.columns].values
    return out
