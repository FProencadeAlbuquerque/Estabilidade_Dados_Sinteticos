"""Step 0 - Monte Carlo generation of IEEE 14-bus load-margin datasets.

Python reproduction of the MATLAB/MATPOWER script that generated `Bus14.csv`: the
active and reactive powers of the monitored loads are drawn from independent normal
distributions centred at their base-case values with a standard deviation of 30 % of
the base value (P7: zero mean, unit standard deviation), the power flow is solved, and
the load margin of bus 14 is the active power added to it up to the nose point of a
proportional increase of all loads and generation (see `sdlm/network.py`).

    --outage 2 3    takes branch 2-3 out of service (topology used in the power flow
                    AND in the continuation, unlike the original contingency file)
    --check         regenerates the intact case and compares it with Bus14.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sdlm import config  # noqa: E402
from sdlm.network import GROWTH, Network  # noqa: E402

LOADS = ["P4", "Q4", "P5", "P7", "P9", "P10", "Q10", "P11", "P12", "Q12", "P13", "P14", "Q14"]
COLUMNS = ["V2", "V6", "V9", "Theta2", "Theta6", "Theta9", "P1", *LOADS, "LoadMargin"]


def generate(n, outage=(), seed=config.SEED, sigma=0.3):
    net = Network("case14", outage)
    rng = np.random.default_rng(seed)
    rows, tried = [], 0
    while len(rows) < n:
        tried += 1
        pd_, qd = net.pd0.copy(), net.qd0.copy()
        draw = {}
        for c in LOADS:
            b = int(c[1:]) - 1
            base = (pd_ if c[0] == "P" else qd)[b]
            v = rng.normal(0.0, 1.0) if c == "P7" else rng.normal(base, sigma * abs(base))
            (pd_ if c[0] == "P" else qd)[b] = v
            draw[c] = v
        V, ok, _ = net.solve(pd_, qd, net.pg0)
        if not ok:
            continue                                   # infeasible draw: resampled
        lam = net.lambda_max(pd_, qd, net.pg0, V)
        rows.append({"V2": abs(V[1]), "V6": abs(V[5]), "V9": abs(V[8]),
                     "Theta2": np.angle(V[1], deg=True), "Theta6": np.angle(V[5], deg=True),
                     "Theta9": np.angle(V[8], deg=True), "P1": net.slack_p(V, pd_), **draw,
                     "LoadMargin": GROWTH * lam * pd_[13]})
    df = pd.DataFrame(rows, columns=COLUMNS)
    return df.round({"V2": 12, "V6": 12}), tried      # set-point buses: drop 1e-16 noise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outage", nargs=2, type=int, action="append", default=[])
    p.add_argument("-n", type=int, default=10_000)
    p.add_argument("--out", default=None)
    p.add_argument("--check", action="store_true")
    a = p.parse_args()
    df, tried = generate(a.n, [tuple(o) for o in a.outage])
    print(f"{len(df)} records ({tried - len(df)} non-convergent draws resampled)")
    if a.check:
        ref = pd.read_csv(config.DATASETS["ieee14"])
        print(pd.DataFrame({"generated_mean": df.mean(), "Bus14_mean": ref[COLUMNS].mean(),
                            "generated_std": df.std(), "Bus14_std": ref[COLUMNS].std()}).round(4).to_string())
    if a.out:
        df.to_csv(a.out, index=False)
        print("written", a.out)
