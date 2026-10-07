"""Aggregate per-seed results into paper-ready CSVs (mean and 95% CI).

Usage:  python -m sim.aggregate --inp results --out ../paper/data
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

W_WINDOW = 0.84   # tau_acc + 3*delta (s)


def mci(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan
    if x.size == 1:
        return x.mean(), 0.0
    h = stats.t.ppf(0.975, x.size - 1) * x.std(ddof=1) / np.sqrt(x.size)
    return x.mean(), h


def agg(df, by, cols, scale=1.0):
    rows = []
    for key, g in df.groupby(by, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        r = dict(zip(by, key))
        r["runs"] = len(g)
        for c in cols:
            m, h = mci(g[c] * scale)
            r[c] = m
            r[c + "_ci"] = h
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="results")
    ap.add_argument("--out", default="results/agg")
    a = ap.parse_args()
    I = lambda f: os.path.join(a.inp, f)
    O = lambda f: os.path.join(a.out, f)
    os.makedirs(a.out, exist_ok=True)

    lat = pd.read_csv(I("latency_runs.csv"))
    t = agg(lat, ["system", "n"], ["p50", "p95", "p99"], 1000)
    t.round(1).to_csv(O("latency_table.csv"), index=False)

    sw = pd.read_csv(I("vdf_sweep_runs.csv"))
    s = agg(sw, ["t_vdf"], ["p50", "p99"], 1000)
    s["t_vdf_ms"] = s["t_vdf"] * 1000
    s["a_max"] = s["t_vdf"] / W_WINDOW
    s.round(3).to_csv(O("vdf_sweep.csv"), index=False)

    br = pd.read_csv(I("latency_breakdown_runs.csv"))
    rows = []
    for c in ["access", "fill", "certify", "await_cut", "agreement", "vdf_order", "cut_interval", "agreement_median", "w_emp_p50", "w_emp_max", "ref_spread_max", "acc_lag_max", "view_changes"]:
        m, h = mci(br[c] * 1000)
        rows.append(dict(component=c, mean_ms=m, ci_ms=h))
    pd.DataFrame(rows).round(1).to_csv(O("latency_breakdown.csv"), index=False)

    ld = pd.read_csv(I("load_runs.csv"))
    l = agg(ld, ["system", "load"], ["p50", "p99"], 1000)
    l["load_k"] = l["load"] / 1000
    l.round(2).to_csv(O("load_long.csv"), index=False)
    # wide format for pgfplots, cut at saturation (p50 > 5 s)
    for sysname, g in l.groupby("system"):
        g = g.sort_values("load")
        g = g[g["p50"] <= 5000]
        fname = "load_" + sysname.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("+", "") + ".csv"
        g[["load_k", "p50", "p50_ci", "p99"]].round(2).to_csv(O(fname), index=False)

    mb = pd.read_csv(I("microblock_runs.csv"))
    agg(mb, ["load", "mb_kib"], ["p50", "p99", "mean_fill_txs", "certify_ms"], 1).assign(
        p50=lambda d: d.p50 * 1000, p50_ci=lambda d: d.p50_ci * 1000,
        p99=lambda d: d.p99 * 1000, p99_ci=lambda d: d.p99_ci * 1000).round(1).to_csv(O("microblock.csv"), index=False)

    j = pd.read_csv(I("jain_runs.csv"))
    agg(j, ["workload"], ["jain_FIFO", "jain_LeaderBFT", "jain_DAGBFT", "jain_ChronoSeq"]).round(4).to_csv(O("jain.csv"), index=False)

    inv = pd.read_csv(I("inversion_runs.csv"))
    ia = agg(inv, ["workload", "system", "gap_lo", "gap_hi"], ["inversion"])
    ia.round(4).to_csv(O("inversion_long.csv"), index=False)
    u = ia[ia.workload == "uniform"].copy()
    u["gap_mid"] = np.sqrt(np.maximum(u.gap_lo, 1) * u.gap_hi)
    wide = u.pivot(index="gap_mid", columns="system", values="inversion").reset_index()
    wide.columns = [c.replace("+", "").replace(" ", "").replace("-", "") for c in wide.columns]
    wide.round(4).to_csv(O("inversion_uniform.csv"), index=False)

    rc = pd.read_csv(I("race_runs.csv"))
    agg(rc, ["mode", "system", "searcher"], ["win"]).round(3).to_csv(O("race.csv"), index=False)

    fr = pd.read_csv(I("frontrun_runs.csv"))
    fa = agg(fr, ["system", "f"], ["success", "success_honest_ingress"])
    fa.round(4).to_csv(O("frontrun_long.csv"), index=False)
    clean = lambda c: c.replace("+", "").replace(" ", "").replace("(", "").replace(")", "").replace("-", "").replace("_", "")
    for col, fname in (("success", "frontrun.csv"), ("success_honest_ingress", "frontrun_honest.csv")):
        wide = fa[~fa.system.str.startswith("_")].pivot(index="f", columns="system", values=col).reset_index()
        wide.columns = [clean(c) if c != "f" else "f" for c in wide.columns]
        wide.round(4).to_csv(O(fname), index=False)

    ce = pd.read_csv(I("censor_runs.csv"))
    agg(ce, ["f"], ["tgt_p50", "tgt_p99", "tgt_max", "tgt_included", "other_p50", "other_p99"]).round(3).to_csv(O("censor.csv"), index=False)

    wh = pd.read_csv(I("withhold_runs.csv"))
    agg(wh, ["f"], ["p50", "p95", "p99", "vc_frac", "cut_interval"]).round(3).to_csv(O("withhold.csv"), index=False)

    co = pd.read_csv(I("consistency_runs.csv"))
    co["vc_rate"] = co.view_changes / co.epochs
    agg(co, ["sigma", "spike", "f_eq"], ["local_div_epochs", "local_incons_vertices", "cut_div_epochs", "vc_rate", "p50", "p99"]).round(4).to_csv(O("consistency.csv"), index=False)
    print("aggregated ->", a.out)


if __name__ == "__main__":
    main()
