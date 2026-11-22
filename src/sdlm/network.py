"""Dense Newton-Raphson power flow and maximum-loading (nose point) search on the
MATPOWER case14 / case118 networks (data from PYPOWER, identical to MATPOWER).

The load margin reproduces the MATLAB scripts that generated the datasets
(`runcpf` with `cpf.stop_at = 'NOSE'`): every load (P and Q) and every generator
active power is scaled by (1 + 0.8*lambda) from the operating point, generator
reactive limits are not enforced, and

    LoadMargin = PD_bus(lambda_max) - PD_bus(0) = 0.8 * lambda_max * PD_bus .

lambda_max is located by bisection on the convergence of the power flow, each
solution being started from the last converged one (upper branch of the PV curve).
"""
import numpy as np

BUS_I, BUS_TYPE, PD, QD, VM, VA = 0, 1, 2, 3, 7, 8
GEN_BUS, PG, QG, VG, GEN_STATUS = 0, 1, 2, 5, 7
GROWTH = 0.8          # target case = 1.8 x base case


class Network:
    def __init__(self, case: str):
        from pypower import case14, case118
        from pypower.ext2int import ext2int
        from pypower.makeYbus import makeYbus

        ppc = ext2int({"case14": case14.case14, "case118": case118.case118}[case]())
        self.baseMVA, bus, gen = ppc["baseMVA"], ppc["bus"], ppc["gen"]
        self.n = bus.shape[0]
        self.Y = np.asarray(makeYbus(self.baseMVA, bus, ppc["branch"])[0].todense())
        self.pd0, self.qd0 = bus[:, PD].copy(), bus[:, QD].copy()
        on = gen[:, GEN_STATUS] > 0
        self.gbus = gen[on, GEN_BUS].astype(int)
        self.pg0 = gen[on, PG].copy()
        self.ext = {int(b): i for i, b in enumerate(ppc["order"]["bus"]["e2i"].nonzero()[0])} \
            if False else {int(e): int(i) for i, e in enumerate(ppc["order"]["bus"]["i2e"])}
        self.gidx = {}
        for k, b in enumerate(self.gbus):
            self.gidx.setdefault(int(b), k)
        self.ref = np.where(bus[:, BUS_TYPE] == 3)[0]
        self.pv = np.where(bus[:, BUS_TYPE] == 2)[0]
        self.pq = np.where(bus[:, BUS_TYPE] == 1)[0]
        self.pvpq = np.r_[self.pv, self.pq]
        vm = np.ones(self.n)
        vm[self.gbus] = gen[on, VG]
        vm[self.pq] = 1.0
        self.V0 = vm * np.exp(1j * np.deg2rad(bus[self.ref[0], VA]))
        self.va_ref = bus[self.ref[0], VA]

    # ---------------------------------------------------------------- power flow
    def sbus(self, pd, qd, pg):
        s = -(pd + 1j * qd)
        np.add.at(s, self.gbus, pg)
        return s / self.baseMVA

    def newton(self, sbus, V, tol=1e-8, max_it=20):
        """Returns (V, converged). Generator reactive limits are not enforced."""
        Y, pvpq, pq = self.Y, self.pvpq, self.pq
        npvpq = len(pvpq)
        for _ in range(max_it):
            mis = V * np.conj(Y @ V) - sbus
            F = np.r_[mis[pvpq].real, mis[pq].imag]
            if np.max(np.abs(F)) < tol:
                return V, True
            I = Y @ V
            Vn = V / np.abs(V)
            dS_dVm = V[:, None] * np.conj(Y * Vn[None, :]) + np.diag(np.conj(I) * Vn)
            dS_dVa = 1j * V[:, None] * np.conj(np.diag(I) - Y * V[None, :])
            J = np.block([[dS_dVa[np.ix_(pvpq, pvpq)].real, dS_dVm[np.ix_(pvpq, pq)].real],
                          [dS_dVa[np.ix_(pq, pvpq)].imag, dS_dVm[np.ix_(pq, pq)].imag]])
            try:
                dx = np.linalg.solve(J, -F)
            except np.linalg.LinAlgError:
                return V, False
            va, vm = np.angle(V), np.abs(V)
            va[pvpq] += dx[:npvpq]
            vm[pq] += dx[npvpq:]
            if not np.all(np.isfinite(vm)) or vm.min() <= 0.05 or vm.max() > 5:
                return V, False
            V = vm * np.exp(1j * va)
        return V, False

    def solve(self, pd, qd, pg, V=None):
        s = self.sbus(pd, qd, pg)
        V, ok = self.newton(s, self.V0.copy() if V is None else V)
        return V, ok, s

    def gen_q(self, V, qd, bus):
        """Reactive output (MVAr) of the generator(s) at internal bus `bus`."""
        return (V[bus] * np.conj(self.Y[bus] @ V)).imag * self.baseMVA + qd[bus]

    def slack_p(self, V, pd):
        b = self.ref[0]
        return (V[b] * np.conj(self.Y[b] @ V)).real * self.baseMVA + pd[b]

    # ---------------------------------------------------------------- nose point
    def lambda_max(self, pd, qd, pg, V, lam_hi=12.0, rel_tol=5e-4):
        """Largest lambda for which the power flow of the scaled case converges."""
        def at(lam, Vs):
            k = 1 + GROWTH * lam
            return self.newton(self.sbus(pd * k, qd * k, pg * k), Vs, max_it=15)

        lo, Vlo, step = 0.0, V, 0.5
        hi = None
        while hi is None and lo < lam_hi:            # expand with warm starts
            Vt, ok = at(lo + step, Vlo)
            if ok:
                lo, Vlo = lo + step, Vt
            else:
                hi = lo + step
        if hi is None:
            return np.nan
        while hi - lo > rel_tol * max(hi, 1e-6):
            mid = 0.5 * (lo + hi)
            Vt, ok = at(mid, Vlo)
            if ok:
                lo, Vlo = mid, Vt
            else:
                hi = mid
        return lo
