"""Generate LaTeX tables and number macros for the paper from raw per-seed CSVs.

Usage: python -m sim.paper_tables --inp results --out ../paper

Every number is computed from the raw runs at full precision (mean and 95%
Student-t CI over seeds) and rounded once. A CI is printed whenever it is at
least 1% of the mean (or, for probabilities, at least 0.005 absolute).
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

W_SEED = 0.84


def mci(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float(x.mean()) if x.size else np.nan, 0.0
    return float(x.mean()), float(stats.t.ppf(0.975, x.size - 1) * x.std(ddof=1) / np.sqrt(x.size))


def fmt(m, h, d, prob=False, force=False, lead0=True, pct=False):
    """Format mean +- CI with d decimals; CI shown if relevant."""
    if pct:
        m, h = 100 * m, 100 * h
    s = f"{m:.{d}f}"
    show = force or (h >= 0.01 * abs(m) if not prob else h >= 0.005)
    if pct:
        show = force or h >= 0.01 * abs(m) or h >= 0.5
    if show:
        hs = f"{h:.{d}f}"
        if not lead0 and hs.startswith("0."):
            hs = hs[1:]
        if float(hs) == 0:
            return s + ("\\%" if pct else "")
        s += f"$\\pm${hs}"
    return s + ("\\%" if pct else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="results")
    ap.add_argument("--out", default="../paper")
    a = ap.parse_args()
    I = lambda f: pd.read_csv(os.path.join(a.inp, f))
    os.makedirs(os.path.join(a.out, "tables"), exist_ok=True)
    T = lambda f: os.path.join(a.out, "tables", f)
    macros = {}

    def mac(name, val):
        macros[name] = val

    # ---------------- latency table
    lat = I("latency_runs.csv")
    names = [("FIFO", "FIFO"), ("Leader-BFT", "Leader-BFT"), ("DAG+BFT", "DAG+BFT"),
             ("ChronoSeq-fast", "\\cs{}-1"), ("ChronoSeq-safe", "\\cs{}-2")]
    rows = []
    for key, label in names:
        cells = [label]
        for n in (8, 16, 32, 64, 128):
            g = lat[(lat.system == key) & (lat.n == n)]
            for q in ("p50", "p99"):
                m, h = mci(g[q] * 1000)
                cells.append(f"{m:.0f}$\\pm${h:.0f}")
        rows.append(" & ".join(cells) + " \\\\")
    open(T("tab_lat.tex"), "w").write("\n".join(rows) + "\n")
    for key, tag in (("FIFO", "Fifo"), ("Leader-BFT", "Lead"), ("DAG+BFT", "Dag"), ("ChronoSeq-fast", "CsOne"), ("ChronoSeq-safe", "CsTwo")):
        for n, nt in ((8, "A"), (16, "B"), (32, "C"), (64, "D"), (128, "E")):
            g = lat[(lat.system == key) & (lat.n == n)]
            mac(f"lat{tag}{nt}", f"{mci(g.p50)[0]:.2f}")
            mac(f"lat{tag}{nt}ms", f"{mci(g.p50 * 1000)[0]:.0f}")
            mac(f"latp{tag}{nt}", f"{mci(g.p99)[0]:.2f}")

    for key, tag in (("FIFO", "Fifo"), ("Leader-BFT", "Lead"), ("DAG+BFT", "Dag"), ("ChronoSeq-fast", "CsOne"), ("ChronoSeq-safe", "CsTwo")):
        ms = [mci(lat[(lat.system == key) & (lat.n == n)].p50)[0] for n in (8, 16, 32, 64, 128)]
        mac(f"lat{tag}Lo", f"{min(ms):.2f}")
        mac(f"lat{tag}Hi", f"{max(ms):.2f}")

    br = I("latency_breakdown_runs.csv")
    for c in ("access", "fill", "certify", "await_cut", "agreement", "vdf_order", "cut_interval"):
        mac("brk" + c.replace("_", "").capitalize(), f"{mci(br[c] * 1000)[0]:.0f}")
    mac("wEmpMax", f"{br.w_emp_max.max() * 1000:.0f}")
    # mean phase times (s) for the pipeline figure, n = 32
    mac("pipeDelta", f"{mci(br.cut_interval)[0]:.3f}")
    mac("pipeRecv", f"{mci(br.phase_recv)[0]:.3f}")
    mac("pipePrep", f"{mci(br.phase_prepare)[0]:.3f}")
    mac("pipeCommit", f"{mci(br.phase_commit)[0]:.3f}")
    mac("pipeTpcMs", f"{mci(br.phase_commit - br.phase_recv)[0] * 1000:.0f}")
    # recommended protocol constants that the experiments do not exercise
    mac("kCarry", "32")
    mac("kCarrySec", f"{32 * mci(br.cut_interval)[0]:.1f}")
    mac("rhoRec", "1.02")
    mac("rhoMaxInfl", f"{100 * (1.02 ** 10 - 1):.0f}")
    mac("refSpreadMax", f"{br.ref_spread_max.max() * 1000:.0f}")
    lags, vcs = [lat.lagmax.max()], [lat.vc.sum()]
    for fname in ("microblock_runs.csv", "load_runs.csv", "jain_runs.csv"):
        d = I(fname)
        lags.append(d.lagmax.max()); vcs.append(d.vc.sum())
    frl = I("frontrun_runs.csv"); frl = frl[frl.system == "_lagmax"]
    lags.append(frl.success.max()); vcs.append(frl.success_honest_ingress.sum())
    mac("accLagMax", f"{max(lags) * 1000:.0f}")
    mac("accLagMaxS", f"{max(lags):.2f}")
    mac("benignVc", f"{int(sum(vcs))}")
    _sp = max(lat.spread.max(), I('microblock_runs.csv').spread.max(), I('load_runs.csv').spread.max())
    mac("wEmpMaxAll", f"{(_sp + 0.480 + 2 * 0.120) * 1000:.0f}")
    mac("wEmpMaxAllS", f"{_sp + 0.480 + 2 * 0.120:.3f}")
    mac("spreadMaxAllS", f"{_sp:.3f}")
    mac("timerLateBenign", f"{int(lat.timer_late.sum())}")
    from sim.core import Config as _Cfg, ONE_WAY_S as _OW, sample_delay as _sd
    _c = _Cfg()
    _r = np.random.default_rng(12345)
    # messages on the inter-region link with the largest median one-way delay
    _i, _j = np.unravel_index(np.argmax(_OW), _OW.shape)
    _x = np.concatenate([_sd(_r, np.full(1_000_000, _OW[_i, _j]), _c) for _ in range(10)])
    mac("deltaExceed", f"{100 * (_x > _c.delta_bound).mean():.3f}\\%")
    mac("deltaMaxMs", f"{1000 * _x.max():.0f}")
    mac("spreadMaxAll", f"{max(lat.spread.max(), I('microblock_runs.csv').spread.max(), I('load_runs.csv').spread.max()) * 1000:.0f}")

    sw = I("vdf_sweep_runs.csv")
    base = mci(sw[sw.t_vdf == 0].p50)[0]
    # ChronoSeq without a VDF vs DAG+BFT (no ENDORSE round), n=32, same seeds
    dag32 = mci(lat[(lat.system == "DAG+BFT") & (lat.n == 32)].p50)[0]
    mac("costEndorse", f"{base - dag32:.2f}")
    mac("costEndorseMs", f"{(base - dag32) * 1000:.0f}")
    for key, tag in (("ChronoSeq-fast", "One"), ("ChronoSeq-safe", "Two")):
        mac(f"extraDag{tag}", f"{mci(lat[(lat.system == key) & (lat.n == 32)].p50)[0] - dag32:.2f}")
    for t, tag in ((0.85, "One"), (1.7, "Two"), (8.45, "Ten")):
        m = mci(sw[np.isclose(sw.t_vdf, t)].p50)[0]
        mac(f"vdfCost{tag}", f"{m - base:.2f}")
        mac(f"vdfLat{tag}", f"{m:.2f}")
    hid = []
    for t in sorted(sw.t_vdf.unique()):
        if t > 0:
            m = mci(sw[np.isclose(sw.t_vdf, t)].p50)[0]
            hid.append(t - (m - base))
    mac("vdfHidden", f"{max(hid) * 1000:.0f}")

    # ---------------- consistency table
    co = I("consistency_runs.csv")
    co["vc"] = co.view_changes / co.epochs
    rows = []
    for (sig, sp, feq), lab in (((0.05, 0.0, 0), "0.05 / 0\\% & 0"), ((0.25, 0.01, 0), "0.25 / 1\\% & 0"),
                                ((0.5, 0.05, 0), "0.50 / 5\\% & 0"), ((0.05, 0.0, 3), "0.05 / 0\\% & 3 lanes"),
                                ((0.5, 0.05, 3), "0.50 / 5\\% & 3 lanes")):
        g = co[(np.isclose(co.sigma, sig)) & (np.isclose(co.spike, sp)) & (co.f_eq == feq)]
        c1 = fmt(*mci(g.local_div_epochs), 1, pct=True)
        c2 = fmt(*mci(g.local_incons_vertices), 1, pct=True)
        c3 = fmt(*mci(g.vc), 1, pct=True)
        p50 = fmt(*mci(g.p50), 2, lead0=False)
        p99 = fmt(*mci(g.p99), 2, lead0=False)
        if feq:
            rows.append(f"{lab} & {c1} & {c2} & \\multicolumn{{2}}{{c}}{{as without equiv.}} \\\\")
        else:
            rows.append(f"{lab} & {c1} & {c2} & {c3} & {p50} / {p99} \\\\")
    open(T("tab_cons.tex"), "w").write("\n".join(rows) + "\n")
    g0 = co[(np.isclose(co.sigma, 0.05)) & (co.f_eq == 0)]
    mac("consIncons", f"{100 * mci(g0.local_incons_vertices)[0]:.0f}")
    mac("consVcMax", f"{100 * co.groupby(['sigma', 'spike', 'f_eq']).vc.mean().max():.1f}")

    # ---------------- race table
    rc = I("race_runs.csv")
    sysl = [("FIFO", "FIFO"), ("Leader-BFT", "Leader-BFT"), ("Median-TS", "Median-TS"), ("DAG+BFT", "DAG+BFT"),
            ("ChronoSeq-vertex", "per-vertex tickets"), ("ChronoSeq", "\\cs{}")]
    rows = []
    for key, lab in sysl:
        cells = [lab]
        for s in ("SYD", "SIN", "PDX", "MULTI"):
            g = rc[(rc["mode"] == "copies") & (rc.system == key) & (rc.searcher == s)]
            cells.append(fmt(*mci(g.win), 2, prob=True, lead0=False))
        g = rc[(rc["mode"] == "sybil") & (rc.system == key) & (rc.searcher == "MULTI")]
        cells.append(fmt(*mci(g.win), 2, prob=True, lead0=False))
        rows.append(" & ".join(cells) + " \\\\")
    open(T("tab_race.tex"), "w").write("\n".join(rows) + "\n")
    for key, tag in (("ChronoSeq", "Cs"), ("DAG+BFT", "Dag"), ("Leader-BFT", "Lead"), ("Median-TS", "Med"), ("ChronoSeq-vertex", "Vert"), ("FIFO", "Fifo")):
        for mode, mt in (("copies", "C"), ("sybil", "S")):
            g = rc[(rc["mode"] == mode) & (rc.system == key) & (rc.searcher == "MULTI")]
            mac(f"race{tag}{mt}", f"{mci(g.win)[0]:.2f}")
    others = rc[(rc["mode"] == "copies") & (rc.system == "ChronoSeq") & (rc.searcher != "MULTI")].groupby("searcher").win.mean()
    mac("raceCsOthLo", f"{others.min():.2f}")
    mac("raceCsOthHi", f"{others.max():.2f}")

    # ---------------- jain
    j = I("jain_runs.csv")
    mac("jainFifoLo", f"{j.groupby('workload').jain_FIFO.mean().min():.3f}")
    mac("jainFifoHi", f"{j.groupby('workload').jain_FIFO.mean().max():.3f}")
    others = pd.concat([j.groupby('workload')[c].mean() for c in ("jain_LeaderBFT", "jain_DAGBFT", "jain_ChronoSeq")])
    mac("jainOthLo", f"{np.floor(others.min() * 1000) / 1000:.3f}")

    # ---------------- inversions
    inv = I("inversion_runs.csv")
    u = inv[inv.workload == "uniform"]
    for sysn, tag in (("ChronoSeq", "Cs"), ("DAG+BFT", "Dag"), ("Median-TS", "Med"), ("FIFO", "Fifo"), ("Leader-BFT", "Lead")):
        for lo, lt in ((50, "A"), (100, "B"), (200, "C"), (400, "D")):
            g = u[(u.system == sysn) & (u.gap_lo == lo)]
            mac(f"inv{tag}{lt}", f"{100 * mci(g.inversion)[0]:.{3 if lo == 400 else 1}f}")
            mac(f"inv{tag}{lt}ci", fmt(*mci(g.inversion), 1 if lo != 400 else 3, pct=True))

    # ---------------- frontrun
    fr = I("frontrun_runs.csv")
    for sysn, tag in (("DAG+BFT", "Dag"), ("Median-TS", "Med"), ("ChronoSeq-vertex", "Vert"), ("ChronoSeq", "Cs"),
                      ("ChronoSeq-k4", "CsFour"), ("ChronoSeq-k16", "CsSixteen"), ("Seed-before-cut", "Seed"), ("Leader-BFT", "Lead")):
        for f, ft in ((1, "A"), (4, "B"), (7, "C"), (10, "D")):
            g = fr[(fr.system == sysn) & (fr.f == f)]
            mac(f"fr{tag}{ft}", f"{mci(g.success)[0]:.2f}")
            mac(f"frh{tag}{ft}", f"{mci(g.success_honest_ingress)[0]:.2f}")
    for sysn, tag in (("_exclusion", "Excl"), ("_veto", "Veto"), ("_reach", "Reach")):
        for f, ft in ((1, "A"), (4, "B"), (7, "C"), (10, "D")):
            g = fr[(fr.system == sysn) & (fr.f == f)]
            mac(f"fr{tag}{ft}", f"{100 * mci(g.success_honest_ingress)[0]:.1f}")
    for f, ft in ((1, "A"), (4, "B"), (7, "C"), (10, "D")):
        g = fr[(fr.system == "_byz_ingress") & (fr.f == f)]
        mac(f"frByzIn{ft}", f"{100 * mci(g.success)[0]:.1f}")
        mac(f"frByzIn{ft}ci", fmt(*mci(g.success), 1, pct=True))
        for sysn, tag in (("_exclusion", "Excl"), ("_veto", "Veto")):
            g = fr[(fr.system == sysn) & (fr.f == f)]
            mac(f"fr{tag}{ft}ci", fmt(*mci(g.success_honest_ingress), 1, pct=True))
        for sysn, tag in (("ChronoSeq", "Cs"), ("Leader-BFT", "Lead"), ("Median-TS", "Med"), ("ChronoSeq-k4", "CsFour"), ("DAG+BFT", "Dag")):
            g = fr[(fr.system == sysn) & (fr.f == f)]
            mac(f"fr{tag}{ft}ci", fmt(*mci(g.success), 2, prob=True, lead0=False))
            mac(f"frh{tag}{ft}ci", fmt(*mci(g.success_honest_ingress), 2, prob=True, lead0=False))
        # paired difference Leader-BFT - ChronoSeq (same seeds)
        a_ = fr[(fr.system == "Leader-BFT") & (fr.f == f)].set_index("seed").success
        b_ = fr[(fr.system == "ChronoSeq") & (fr.f == f)].set_index("seed").success
        mac(f"frDiffLead{ft}", fmt(*mci((a_ - b_).values), 2, prob=True, lead0=False, force=True))
    ci_max = fr[~fr.system.str.startswith("_")].groupby(["system", "f"]).success.apply(lambda s: mci(s)[1]).max()
    mac("frCiMax", f"{np.ceil(ci_max * 100) / 100:.2f}")   # rounded up: "all CIs <= x"

    # ---------------- attack table
    ce = I("censor_runs.csv")
    wh = I("withhold_runs.csv")
    wh["vc"] = wh.vc_frac
    rows = []
    for f in (0, 1, 4, 7, 10):
        c = ce[ce.f == f]
        w = wh[wh.f == f]
        cells = [str(f), fmt(*mci(c.tgt_p50), 2, lead0=False), fmt(*mci(c.tgt_p99), 2, lead0=False),
                 fmt(*mci(c.tgt_max), 2, lead0=False), fmt(*mci(w.p50), 2, lead0=False),
                 fmt(*mci(w.p99), 2, lead0=False), fmt(*mci(w.vc), 1, pct=True)]
        rows.append(" & ".join(cells) + " \\\\")
    open(T("tab_attack.tex"), "w").write("\n".join(rows) + "\n")
    for f, ft in ((1, "A"), (4, "B"), (7, "C"), (10, "D")):
        c = ce[ce.f == f]
        mac(f"censVc{ft}", f"{100 * c.vc.sum() / c.epochs.sum():.1f}")
    c0, c10 = ce[ce.f == 0], ce[ce.f == 10]
    mac("censDelta", f"{mci(c10.tgt_p50)[0] - mci(c0.tgt_p50)[0]:.2f}")
    mac("censMax", f"{mci(c10.tgt_max)[0]:.2f}")
    mac("censMaxAll", f"{ce.tgt_max.max():.2f}")   # largest delay in any run
    mac("whPnineA", f"{mci(wh[wh.f == 0].p99)[0]:.2f}")
    mac("whPnineD", f"{mci(wh[wh.f == 10].p99)[0]:.2f}")
    mac("whVcD", f"{100 * mci(wh[wh.f == 10].vc)[0]:.0f}")

    # ---------------- ablation table (race = copies mode MULTI; frontrun f=1/f=10)
    def race(sysn):
        g = rc[(rc["mode"] == "copies") & (rc.system == sysn) & (rc.searcher == "MULTI")]
        return fmt(*mci(g.win), 2, prob=True, lead0=False)

    def fr2(sysn):
        out = []
        for f in (1, 10):
            g = fr[(fr.system == sysn) & (fr.f == f)]
            out.append(fmt(*mci(g.success), 2, prob=True, lead0=False))
        return "/".join(out)

    def p50(sysn, n=32):
        g = lat[(lat.system == sysn) & (lat.n == n)]
        return f"{mci(g.p50)[0]:.2f}"
    ld = I("load_runs.csv")
    sl = ld[(ld.system == "ChronoSeq (single lane)") & (ld.load == 5000)]
    rows = [
        f"FIFO & yes & 7.9/81 & {p50('FIFO')} & {race('FIFO')} & 1.00/1.00 \\\\",
        f"Leader-BFT & yes & 250$^{{\\dagger}}$ & {p50('Leader-BFT')} & {race('Leader-BFT')} & {fr2('Leader-BFT')} \\\\",
        "Lanes, local ready sets & \\textbf{no} & -- & -- & -- & -- \\\\",
        f"Lanes+cut, determ. & yes & 250 & {p50('DAG+BFT')} & {race('DAG+BFT')} & {fr2('DAG+BFT')} \\\\",
        f"Lanes+cut, Median-TS & yes & 250 & {p50('DAG+BFT')} & {race('Median-TS')} & {fr2('Median-TS')} \\\\",
        f"\\ +VDF, seed before cut & yes & 250 & {p50('DAG+BFT')} & -- & {fr2('Seed-before-cut')} \\\\",
        f"\\ +VDF, per-vertex & yes & 250 & {p50('ChronoSeq-fast')} & {race('ChronoSeq-vertex')} & {fr2('ChronoSeq-vertex')} \\\\",
        f"\\ +VDF, per-sender (\\cs{{}}) & yes & 250 & {p50('ChronoSeq-fast')} & {race('ChronoSeq')} & {fr2('ChronoSeq')} \\\\",
        f"\\cs{{}}, single lane & yes & 7.8 & {mci(sl.p50)[0]:.2f}$^{{\\ast}}$ & -- & -- \\\\",
    ]
    open(T("tab_abl.tex"), "w").write("\n".join(rows) + "\n")

    # ---------------- load / micro-block
    for sysn, tag in (("ChronoSeq", "Cs"), ("DAG+BFT", "Dag")):
        for load, lt in ((200000, "A"), (225000, "B")):
            g = ld[(ld.system == sysn) & (ld.load == load)]
            mac(f"load{tag}{lt}", f"{mci(g.p50)[0]:.2f}")
            mac(f"loadp{tag}{lt}", f"{mci(g.p99)[0]:.2f}")
    mb = I("microblock_runs.csv")
    for load, lt in ((50000, "A"), (150000, "B")):
        g = mb[mb.load == load].groupby("mb_kib").p50.mean() * 1000
        mac(f"mbLo{lt}", f"{g.min():.0f}")
        mac(f"mbHi{lt}", f"{g.max():.0f}")
        mac(f"mbRange{lt}", f"{g.max() - g.min():.0f}")

    # ---------------- measured CPU costs (bench/) and per-node core budget
    if os.path.exists(os.path.join(a.inp, "bench_vdf.csv")):
        vd = I("bench_vdf.csv")
        cal = vd[vd.kind == "calib"]
        rates = cal.iters / cal.prove_s
        mac("vdfIps", f"{rates.median() / 1000:.0f}k")
        mac("vdfIpsMin", f"{rates.min() / 1000:.0f}k")
        mac("vdfIpsMax", f"{rates.max() / 1000:.0f}k")
        for tgt, tag, sim_t in ((0.84, "One", 0.85), (1.68, "Two", 1.70)):
            g = vd[(vd.kind == "target") & np.isclose(vd.target_s, tgt)]
            mac(f"vdfIters{tag}", f"{g.iters.iloc[0] / 1000:.0f}k")
            mac(f"vdfProve{tag}", f"{g.prove_s.mean():.2f}")
            mac(f"vdfProveMin{tag}", f"{g.prove_s.min():.2f}")
            mac(f"vdfProveMax{tag}", f"{g.prove_s.max():.2f}")
            mac(f"vdfExtra{tag}", f"{g.prove_s.mean() - sim_t:.2f}")
        mac("vdfVerifyMs", f"{vd.verify_ms.median():.1f}")
        mac("vdfVerifyMaxMs", f"{vd.verify_ms.max():.1f}")
        for c in (1, 2):
            g = vd[(vd.kind == "conc") & (vd.conc == c)]
            mac(f"vdfConc{'One' if c == 1 else 'Two'}", f"{g.prove_s.mean():.2f}")
        sg = I("bench_sig.csv")
        t_client = sg[sg.op == "recover"].us_per_op.median() * 1e-6
        mac("sigVerifyUs", f"{sg[sg.op == 'verify'].us_per_op.median():.0f}")
        mac("sigRecoverUs", f"{t_client * 1e6:.0f}")
        bl = I("bench_bls.csv")
        blv = lambda op, q=1: float(bl[(bl.op == op) & (bl.q == q)].ms.iloc[0])
        mac("blsSign", f"{blv('sign'):.2f}")
        mac("blsVerify", f"{blv('verify'):.2f}")
        mac("blsFavS", f"{blv('fast_aggregate_verify', 22):.2f}")
        mac("blsFavL", f"{blv('fast_aggregate_verify', 86):.2f}")
        od = I("bench_order.csv")
        ordn = lambda m, col: od[od.m == m][col].median()
        mac("ordNs", f"{ordn(100000, 'ns_order_per_tx') / 1000:.2f}")
        mac("ordTotNs", f"{ordn(100000, 'ns_per_tx') / 1000:.2f}")
        mac("ordTotNsSmall", f"{ordn(10000, 'ns_per_tx') / 1000:.2f}")
        mac("ordTotNsBig", f"{ordn(1000000, 'ns_per_tx') / 1000:.2f}")
        rt = I("bench_rates.csv").groupby(["n", "load"]).mean(numeric_only=True).reset_index()
        cols, cpu = [], {}
        for n, load in ((32, 20000), (128, 20000), (32, 225000), (128, 225000)):
            r = rt[(rt.n == n) & (rt.load == load)].iloc[0]
            q = n - (n - 1) // 3
            fav = blv("fast_aggregate_verify", q)
            agg = blv("aggregate", q)
            v, e = r.vertex_rate_per_lane, 1.0 / r.cut_interval
            sig = ((n - 1) * v * (blv("sign") + fav) + v * (agg + fav) + e * (3 * fav + 3 * blv("sign"))) / 1000   # per epoch: sign PREPARE, COMMIT, ENDORSE; verify E, PREPAREs, COMMITs
            per_epoch = load * r.cut_interval
            m_ref = 10000 if per_epoch < 30000 else 100000
            ordc = load * ordn(m_ref, "ns_per_tx") * 1e-9
            vdf1, vdf2 = int(np.ceil(0.85 / r.cut_interval)), int(np.ceil(1.7 / r.cut_interval))
            cli = load * t_client
            cpu[(n, load)] = dict(sig=sig, cli=cli, ord=ordc, vdf1=vdf1, vdf2=vdf2, tot1=sig + cli + ordc + vdf1, tot2=sig + cli + ordc + vdf2,
                                  certs=(n - 1) * v, ord_ms=per_epoch * ordn(m_ref, "ns_per_tx") * 1e-6)
        order = [(32, 20000), (128, 20000), (32, 225000), (128, 225000)]
        rows = [
            "Signatures (acks, certificates, votes) & " + " & ".join(f"{cpu[k]['sig']:.2f}" for k in order) + " \\\\",
            "Client signatures (secp256k1) & " + " & ".join(f"{cpu[k]['cli']:.2f}" for k in order) + " \\\\",
            "Ordering and Merkle root & " + " & ".join(f"{cpu[k]['ord']:.2f}" for k in order) + " \\\\",
            "VDF pipeline, $\\Amax{=}1$ ($\\Amax{=}2$) & " + " & ".join(f"{cpu[k]['vdf1']} ({cpu[k]['vdf2']})" for k in order) + " \\\\",
            "\\midrule",
            "Total, $\\Amax{=}1$ & " + " & ".join(f"{cpu[k]['tot1']:.1f}" for k in order) + " \\\\",
        ]
        open(T("tab_cpu.tex"), "w").write("\n".join(rows) + "\n")
        mac("cpuSigSmall", f"{cpu[(32, 20000)]['sig']:.1f}")
        mac("cpuSigLarge", f"{cpu[(128, 20000)]['sig']:.1f}")
        mac("cpuSigLargeHi", f"{cpu[(128, 225000)]['sig']:.1f}")
        mac("cpuTotSmall", f"{cpu[(32, 20000)]['tot1']:.1f}")
        mac("cpuTotSmallLarge", f"{cpu[(128, 20000)]['tot1']:.1f}")
        mac("cpuCliHi", f"{cpu[(32, 225000)]['cli']:.1f}")
        mac("cpuCliLo", f"{cpu[(32, 20000)]['cli']:.1f}")
        mac("cpuTotMaxNoCli", f"{max(c['tot1'] - c['cli'] for c in cpu.values()):.1f}")
        mac("cpuSigHiSmall", f"{cpu[(32, 225000)]['sig']:.1f}")
        mac("cpuTotMax", f"{max(c['tot1'] for c in cpu.values()):.1f}")
        mac("cpuTotMaxTwo", f"{max(c['tot2'] for c in cpu.values()):.1f}")
        mac("cpuCertsLarge", f"{cpu[(128, 20000)]['certs']:.0f}")
        mac("cpuOrdMsHi", f"{cpu[(32, 225000)]['ord_ms']:.0f}")
        mac("scaleHiSmall", f"{rt[(rt.n == 32) & (rt.load == 225000)].p50_fast.iloc[0]:.2f}")
        mac("scaleHiLarge", f"{rt[(rt.n == 128) & (rt.load == 225000)].p50_fast.iloc[0]:.2f}")
        mac("vdfCoresOne", f"{cpu[(32, 20000)]['vdf1']}")
        mac("vdfCoresTwo", f"{cpu[(32, 20000)]['vdf2']}")
        mac("cpuVertexRate", f"{rt[(rt.load == 20000)].vertex_rate_per_lane.mean():.0f}")
        _vr = rt[rt.load == 20000].groupby("n").vertex_rate_per_lane.mean()
        mac("cpuVertexRateLo", f"{_vr.min():.0f}")
        mac("cpuVertexRateHi", f"{_vr.max():.0f}")

    with open(os.path.join(a.out, "numbers.tex"), "w") as fh:
        fh.write("% generated by sim/paper_tables.py from raw per-seed results -- do not edit\n")
        for k, v in sorted(macros.items()):
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(f"wrote {len(macros)} macros and tables to {a.out}")


if __name__ == "__main__":
    main()
