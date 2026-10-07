"""Experiments for the revised ChronoSeq evaluation.

Usage:
    python -m sim.experiments --out results/ --seeds 10 [--only NAME ...]

Every experiment is repeated over independent seeds; CSVs contain one row per
(configuration, seed) and are aggregated (mean and 95% CI) by
``sim.aggregate``.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import replace
from multiprocessing import Pool

import numpy as np
import pandas as pd

from .core import (Config, ONE_WAY_S, append_txs, gen_workload, mix64,
                   node_regions, run_fifo, sample_delay)
from .runner import (GAP_BINS_MS, inversion_by_gap, jain, leader_order_rank,
                     median_ts, order_rank, sender_tickets, simulate_dag,
                     simulate_leader_bft)

W_SEED = 0.84                        # tau_acc + 3*delta = 0.48 + 3*0.12 (Theorem 3)
T_FAST, T_SAFE = 0.85, 1.70          # smallest 50 ms grid values > A_max*W, A_max = 1, 2
PCTS = (50, 95, 99)


def pct(x, q):
    """Percentile that keeps never-sequenced txs (inf) in the population."""
    x = x[~np.isnan(x)]
    return float(np.percentile(x, q, method="lower")) if x.size else np.nan


def steady(run_or_w, cfg):
    w = run_or_w if hasattr(run_or_w, "submit") else run_or_w.w
    return (w.submit >= cfg.warmup) & (w.submit <= cfg.duration - 0.5)


# ---------------------------------------------------------------------------
# E1  consistency: agreed cut vs naive local ready sets
# ---------------------------------------------------------------------------

def baseline_run(cfg, seed, run, **kw):
    """Same workload and lanes, agreement without the ENDORSE round (DAG+BFT, Median-TS)."""
    # same seed: the workload, Byzantine set, ingress and lanes are identical to `run`
    return simulate_dag(replace(cfg, endorse=False), seed, **kw)


def exp_consistency(args):
    (sig, spike, f_eq), seed = args
    cfg = Config(duration=15, jitter_sigma=sig, spike_prob=spike)
    run = simulate_dag(cfg, seed)
    L, byz = run.L, run.byz
    rng = np.random.default_rng(seed + 991)
    honest = ~byz
    # naive local-ready-set baseline: each node orders at epoch boundaries
    # k*Delta the vertices it has locally received
    delta = float(np.median(np.diff(run.Ep.P)))
    A = L.avail[:, honest]
    ok = np.isfinite(A).all(axis=1) & (L.created > cfg.warmup) & (L.created < cfg.duration)
    k = np.ceil(A[ok] / delta)
    inconsistent = k.min(axis=1) != k.max(axis=1)
    eq_lanes = rng.choice(cfg.n, f_eq, replace=False) if f_eq else np.array([], dtype=int)
    inconsistent |= np.isin(L.lane[ok], eq_lanes)
    kk = k.max(axis=1)
    epochs = np.unique(kk)
    bad = np.unique(kk[inconsistent])
    lat = run.latency(T_FAST)[steady(run, cfg)]
    return dict(sigma=sig, spike=spike, f_eq=f_eq, seed=seed,
                local_div_epochs=len(bad) / max(1, len(epochs)),
                local_incons_vertices=float(inconsistent.mean()),
                cut_div_epochs=0.0, view_changes=int((run.Ep.views > 0).sum()),
                lock_fails=int(run.Ep.lock_fails), timer_late=int(run.Ep.timer_late.sum()),
                epochs=len(run.Ep.P), p50=pct(lat, 50), p99=pct(lat, 99))


# ---------------------------------------------------------------------------
# E2  latency comparison, breakdown, VDF sweep
# ---------------------------------------------------------------------------

VDF_SWEEP = (0.0, 0.15, 0.30, 0.45, 0.85, 1.30, 1.70, 2.55, 4.25, 8.45)


def exp_latency(args):
    (n,), seed = args
    cfg = Config(n=n)
    run = simulate_dag(cfg, seed)
    run_b = baseline_run(cfg, seed, run)
    m = steady(run, cfg)
    rows = []
    for name, r_, T in (("DAG+BFT", run_b, 0.0), ("ChronoSeq-fast", run, T_FAST), ("ChronoSeq-safe", run, T_SAFE)):
        lat = r_.latency(T)[m]
        rows.append(dict(system=name, n=n, seed=seed, lagmax=float(run.Ep.lags.max()), vc=int((run.Ep.views > 0).sum()),
                         timer_late=int(r_.Ep.timer_late.sum()),
                         spread=float(r_.Ep.ref_spread.max()), **{f"p{q}": pct(lat, q) for q in PCTS}))
    fr = run_fifo(cfg, run.w, False, np.random.default_rng(seed + 7))
    lat = (fr.seq - run.w.submit)[m]
    rows.append(dict(system="FIFO", n=n, seed=seed, **{f"p{q}": pct(lat, q) for q in PCTS}))
    lr = simulate_leader_bft(run, seed)
    st = lr.Ep.seq_time(cfg, t_vdf=0.0, ntx=lr.ntx_epoch)
    ok = lr.tx_epoch >= 0
    seq = np.full(len(run.w), np.inf)
    seq[ok] = st[lr.tx_epoch[ok], run.ingress[ok]]
    lat = (seq - run.w.submit)[m]
    rows.append(dict(system="Leader-BFT", n=n, seed=seed, **{f"p{q}": pct(lat, q) for q in PCTS}))
    sweep = []
    if n == 32:
        for T in VDF_SWEEP:
            lat = run.latency(T)[m]
            sweep.append(dict(t_vdf=T, seed=seed, p50=pct(lat, 50), p99=pct(lat, 99)))
        # breakdown (ChronoSeq-fast)
        L, Ep, w = run.L, run.Ep, run.w
        v = L.tx_vertex
        e = run.tx_epoch
        ok = m & (e >= 0)
        seq = run.seq(T_FAST)
        parts = dict(
            access=run.arr - w.submit,
            fill=L.created[v] - run.arr,
            certify=L.cert[v] - L.created[v],
            await_cut=Ep.P[np.maximum(e, 0)] - L.cert[v],
            agreement=Ep.cc[np.maximum(e, 0), run.ingress] - Ep.P[np.maximum(e, 0)],
            vdf_order=seq - Ep.cc[np.maximum(e, 0), run.ingress],
        )
        brk = dict(seed=seed, **{k: float(np.mean(val[ok])) for k, val in parts.items()})
        brk["cut_interval"] = float(np.median(np.diff(Ep.P)))
        brk["w_emp_p50"] = float(np.median(Ep.w_emp))
        brk["w_emp_max"] = float(np.max(Ep.w_emp))
        brk["ref_spread_max"] = float(np.max(Ep.ref_spread))
        brk["acc_lag_max"] = float(np.max(Ep.lags))
        brk["view_changes"] = int((Ep.views > 0).sum())
        brk["agreement_median"] = float(np.median(Ep.cc - Ep.P[:, None]))
        # per-epoch phase times at a node, relative to the proposal (for the pipeline figure)
        st = (Ep.P > cfg.warmup) & (Ep.P < cfg.duration)
        brk["phase_recv"] = float(np.mean(Ep.rv[st] - Ep.P[st, None]))
        brk["phase_prepare"] = float(np.mean(Ep.pc[st] - Ep.P[st, None]))
        brk["phase_commit"] = float(np.mean(Ep.cc[st] - Ep.P[st, None]))
    else:
        brk = None
    return dict(rows=rows, sweep=sweep, breakdown=brk)


# ---------------------------------------------------------------------------
# E3  offered load sweep (capacity), E4 micro-block size
# ---------------------------------------------------------------------------

LOADS = (2500, 5000, 7500, 10000, 25000, 50000, 100000, 150000, 200000, 225000)


def exp_load(args):
    (load,), seed = args
    cfg = Config(load_tps=load, duration=8, warmup=2, drain=6)
    rows = []
    run = simulate_dag(cfg, seed)
    run_b = baseline_run(cfg, seed, run)
    m = steady(run, cfg)
    for name, r_, T in (("DAG+BFT", run_b, 0.0), ("ChronoSeq", run, T_FAST)):
        lat = r_.latency(T)[m]
        rows.append(dict(system=name, load=load, seed=seed, p50=pct(lat, 50), p99=pct(lat, 99),
                         lagmax=float(r_.Ep.lags.max()), vc=int((r_.Ep.views > 0).sum()), spread=float(r_.Ep.ref_spread.max())))
    for fan, name in ((3, "FIFO (relay tree)"), (0, "FIFO (direct)")):
        fr = run_fifo(replace(cfg, fifo_fanout=fan), run.w, False, np.random.default_rng(seed + 7))
        lat = (fr.seq - run.w.submit)[m]
        rows.append(dict(system=name, load=load, seed=seed, p50=pct(lat, 50), p99=pct(lat, 99)))
    if load <= 25000:
        cfg1 = replace(cfg, single_lane=True)
        run1 = simulate_dag(cfg1, seed, w=run.w)
        lat = run1.latency(T_FAST)[m]
        rows.append(dict(system="ChronoSeq (single lane)", load=load, seed=seed,
                         p50=pct(lat, 50), p99=pct(lat, 99)))
    return rows


MB_KIB = (16, 32, 64, 128, 256, 512)


def exp_microblock(args):
    (kib, load), seed = args
    cfg = Config(mb_bytes=kib * 1024, load_tps=load, duration=8, warmup=2, drain=6)
    run = simulate_dag(cfg, seed)
    m = steady(run, cfg)
    lat = run.latency(T_FAST)[m]
    return dict(mb_kib=kib, load=load, seed=seed, p50=pct(lat, 50), p99=pct(lat, 99),
                lagmax=float(run.Ep.lags.max()), vc=int((run.Ep.views > 0).sum()), spread=float(run.Ep.ref_spread.max()),
                mean_fill_txs=float(np.mean(run.L.ntx)),
                certify_ms=float(np.median(run.L.cert - run.L.created)) * 1000)


# ---------------------------------------------------------------------------
# E5  fairness metrics (service equity, order inversions)
# ---------------------------------------------------------------------------

def exp_fairness(args):
    (wl,), seed = args
    cfg = Config(workload=wl)
    run = simulate_dag(cfg, seed)
    run_b = baseline_run(cfg, seed, run)
    m = steady(run, cfg)
    w = run.w
    rng = np.random.default_rng(seed + 5)
    fr = run_fifo(cfg, w, False, np.random.default_rng(seed + 7))
    lr = simulate_leader_bft(run, seed)
    st = lr.Ep.seq_time(cfg, t_vdf=0.0, ntx=lr.ntx_epoch)
    okl = lr.tx_epoch >= 0
    seq_l = np.full(len(w), np.inf)
    seq_l[okl] = st[lr.tx_epoch[okl], run.ingress[okl]]
    out = dict(workload=wl, seed=seed, lagmax=float(run.Ep.lags.max()), vc=int((run.Ep.views > 0).sum()),
               jain_FIFO=jain((fr.seq - w.submit)[m], w.client[m]),
               jain_LeaderBFT=jain((seq_l - w.submit)[m], w.client[m]),
               jain_DAGBFT=jain(run_b.latency(0.0)[m], w.client[m]),
               jain_ChronoSeq=jain(run.latency(T_FAST)[m], w.client[m]))
    inv = []
    ranks = {
        "FIFO": np.argsort(np.argsort(fr.order_time, kind="stable"), kind="stable"),
        "Leader-BFT": leader_order_rank(run, lr),
        "DAG+BFT": order_rank(run_b, "det"),
        "Median-TS": order_rank(run_b, "median"),
        "ChronoSeq": order_rank(run, "cs"),
    }
    valid = m & (run.tx_epoch >= 0) & (run_b.tx_epoch >= 0) & okl
    for name, r in ranks.items():
        rates = inversion_by_gap(w.submit, r, valid, rng)
        for (lo, hi), val in zip(zip(GAP_BINS_MS[:-1], GAP_BINS_MS[1:]), rates):
            inv.append(dict(workload=wl, seed=seed, system=name, gap_lo=lo, gap_hi=hi, inversion=val))
    return dict(jain=out, inv=inv)


# ---------------------------------------------------------------------------
# E6  latency race (position neutrality)
# ---------------------------------------------------------------------------

SEARCHERS = ("SYD", "SIN", "PDX", "MULTI")


def race_hook(cfg, sybil: bool):
    """Four searchers react to the same event. MULTI submits one transaction to
    every node: byte-identical copies (sybil=False) or 32 distinct funded
    accounts (sybil=True)."""
    def hook(w, byz, rng):
        ev = np.sort(rng.uniform(cfg.warmup, cfg.duration - 1.0, rng.poisson((cfg.duration - 1 - cfg.warmup) / 0.3)))
        subs, cli, reg, hsh, tag, hint = [], [], [], [], [], []
        base_c = 10_000_000
        nreg = node_regions(cfg.n)
        for k, t0 in enumerate(ev):
            for s, name in enumerate(SEARCHERS):
                h = np.uint64(rng.integers(0, 2**63 - 1))
                c = base_c + k * 200 + s * 40
                if name == "MULTI":
                    for j in range(cfg.n):
                        hj = np.uint64(rng.integers(0, 2**63 - 1)) if sybil else h
                        cj = c + j if sybil else c
                        subs.append(t0 + rng.uniform(0, 0.001)); cli.append(cj); reg.append(nreg[j])
                        hsh.append(hj); tag.append(k * 10 + s + 1); hint.append(j)
                else:
                    subs.append(t0 + rng.uniform(0, 0.001)); cli.append(c); reg.append(s)
                    hsh.append(h); tag.append(k * 10 + s + 1); hint.append(-1)
        return append_txs(w, submit=np.array(subs), client=np.array(cli), region=np.array(reg),
                          txhash=np.array(hsh, dtype=np.uint64), tag=np.array(tag),
                          ingress_hint=np.array(hint))
    return hook


def _winners(rank, sel, ev, who, ok):
    r = rank[sel].astype(float)
    r[~ok[sel]] = np.inf
    best = np.full((ev.max() + 1, len(SEARCHERS)), np.inf)
    np.minimum.at(best, (ev, who), r)
    done = np.isfinite(best).any(axis=1)
    return np.argmin(best[done], axis=1), int(done.sum())


def exp_race(args):
    (mode,), seed = args
    sybil = mode == "sybil"
    cfg = Config()
    run = simulate_dag(cfg, seed, workload_hook=race_hook(cfg, sybil))
    run_b = baseline_run(cfg, seed, run, workload_hook=race_hook(cfg, sybil))
    w = run.w
    sel = w.tag > 0
    ev = (w.tag[sel] - 1) // 10
    who = (w.tag[sel] - 1) % 10
    inc = run.tx_epoch >= 0
    inc_b = run_b.tx_epoch >= 0
    rows = []
    # Pompe-style median timestamps: per node, a searcher's earliest copy
    idx = np.flatnonzero(sel)
    g = run_b.L.tx_vertex[idx]
    A = run_b.L.avail[np.maximum(g, 0)].copy()
    A[g < 0] = np.inf
    key = ev * 10 + who
    uk, inv = np.unique(key, return_inverse=True)
    per_node = np.full((uk.size, cfg.n), np.inf)
    np.minimum.at(per_node, inv, A)
    med = np.median(per_node, axis=1)
    big = np.iinfo(np.int64).max
    ep = np.full(uk.size, big)
    np.minimum.at(ep, inv, np.where(run_b.tx_epoch[idx] >= 0, run_b.tx_epoch[idx], big))
    order = np.lexsort((med, ep))
    mr = np.empty(uk.size)
    mr[order] = np.arange(uk.size)
    med_rank = np.zeros(len(w))
    med_rank[idx] = mr[inv]
    lr = simulate_leader_bft(run, seed)
    policies = [("DAG+BFT", order_rank(run_b, "det"), inc_b),
                ("Median-TS", med_rank, inc_b),
                ("ChronoSeq-vertex", order_rank(run, "cs_vertex"), inc),
                ("ChronoSeq", order_rank(run, "cs"), inc),
                ("Leader-BFT", leader_order_rank(run, lr), lr.tx_epoch >= 0)]
    for name, rank, ok in policies:
        win, nd = _winners(rank, sel, ev, who, ok)
        for s, nm in enumerate(SEARCHERS):
            rows.append(dict(mode=mode, system=name, seed=seed, searcher=nm, win=float((win == s).mean()), events=nd))
    rng = np.random.default_rng(seed + 3)
    E = int(ev.max() + 1)
    arr = np.zeros((E, 4))
    for s in range(3):
        base = ONE_WAY_S[s, cfg.fifo_leader_region] + cfg.access_ms / 1000
        arr[:, s] = rng.uniform(0, 0.001, E) + sample_delay(rng, base, cfg, E)
    arr[:, 3] = rng.uniform(0, 0.001, E) + sample_delay(rng, 0.0005, cfg, E)
    win = np.argmin(arr, axis=1)
    for s, nm in enumerate(SEARCHERS):
        rows.append(dict(mode=mode, system="FIFO", seed=seed, searcher=nm, win=float((win == s).mean()), events=E))
    return rows


# ---------------------------------------------------------------------------
# E7  frontrunning by Byzantine sequencers (targeted injection)
# ---------------------------------------------------------------------------

def exp_frontrun(args):
    (f_byz,), seed = args
    cfg = Config(f_byz=f_byz, duration=15)
    run = simulate_dag(cfg, seed)
    rows = _frontrun_eval(cfg, run, seed, f_byz)
    # DAG+BFT and Median-TS run without the ENDORSE round (same workload and lanes)
    rows_b = {r["system"]: r for r in _frontrun_eval(replace(cfg, endorse=False), baseline_run(cfg, seed, run), seed, f_byz)}
    for r in rows:
        if r["system"] in ("DAG+BFT", "Median-TS"):
            r.update(success=rows_b[r["system"]]["success"],
                     success_honest_ingress=rows_b[r["system"]]["success_honest_ingress"])
    return rows


def _frontrun_eval(cfg, run, seed, f_byz):
    L, Ep, w, byz = run.L, run.Ep, run.w, run.byz
    rng = np.random.default_rng(seed + 11)
    n, q = cfg.n, cfg.q
    need = q - f_byz                     # honest acceptors a faulty leader needs
    nreg = node_regions(n)
    D = ONE_WAY_S[nreg[:, None], nreg[None, :]]
    B = np.flatnonzero(byz)
    honest = ~byz
    e_tx = run.tx_epoch
    cand = np.flatnonzero(steady(run, cfg) & (e_tx >= 0) & (w.submit < cfg.duration - 2))
    victims = rng.choice(cand, min(1000, cand.size), replace=False)
    stick = sender_tickets(run)
    E = len(Ep.P)
    names = ("DAG+BFT", "Median-TS", "ChronoSeq-vertex", "ChronoSeq", "ChronoSeq-k4",
             "ChronoSeq-k16", "Seed-before-cut", "Leader-BFT")
    succ = {k: 0 for k in names}
    succ_h = {k: 0 for k in names}
    n_h = reach = excl_n = veto_n = byz_in = 0
    lr = simulate_leader_bft(run, seed)
    for x in victims:
        g = L.tx_vertex[x]
        i, h, ev = int(L.lane[g]), int(L.height[g]), int(e_tx[x])
        if byz[run.ingress[x]]:
            # a faulty ingress node sees the tx first, holds it for one cut and
            # places its own transaction earlier: success for every design
            byz_in += 1
            for k in names:
                succ[k] += 1
            continue
        n_h += 1
        t_obs = L.avail[g, B].min() + 0.001
        b_first = B[np.argmin(L.avail[g, B])]
        rtt = sample_delay(rng, D[B][:, :, None] * np.ones((1, 1, 2)), cfg).sum(axis=2)
        rtt[np.arange(B.size), B] = 0.0
        certb = t_obs + np.partition(rtt, q - 1, axis=1)[:, q - 1]
        arrive = certb[:, None] + sample_delay(rng, D[B][:, Ep.proposer], cfg)
        okE = Ep.P[None, :] >= arrive
        eb = np.where(okE.any(axis=1), okE.argmax(axis=1), E + 10)
        e_adv = int(eb.min())
        # a faulty leader of the victim's epoch excludes the victim's vertex if
        # enough honest floors allow it and its own tx is already included
        ev_eff = ev
        fs = Ep.floor_sorted[ev]
        if byz[Ep.proposer[ev]] and fs[min(need, fs.shape[0]) - 1, i] <= h and e_adv <= ev:
            ev_eff = ev + 1
            excl_n += 1
        reach += e_adv <= ev_eff
        earlier = e_adv < ev_eff
        same = (e_adv == ev_eff == ev)
        res = {}
        # deterministic order: best Byzantine lane, injected vertex first in lane
        prev = Ep.cut[ev - 1, i] if ev > 0 else 0
        vkey = (h - prev, (i - ev) % n, int(L.tx_pos[x]))
        res["DAG+BFT"] = earlier or (same and any(e_b == ev and (0, (b - ev) % n, 0) < vkey for b, e_b in zip(B, eb)))
        # median receive timestamp; Byzantine nodes report -inf for the attacker
        # and +inf for the victim
        ts_v = np.where(honest, L.avail[g], np.inf)
        ts_a = np.where(honest, t_obs + sample_delay(rng, D[b_first], cfg), -np.inf)
        res["Median-TS"] = earlier or (same and np.median(ts_a) < np.median(ts_v))
        # per-vertex tickets: adversarial roots merged with lane chains
        res["ChronoSeq-vertex"] = earlier or (same and vertex_race(run, ev, g, int((eb == ev).sum()), rng))
        # per-sender tickets with k funded senders. Each view of the victim's
        # epoch led by a faulty node allows one informed veto (Theorem 3(b)):
        # an epoch whose leaders L(t,0), ..., L(t,r-1) are faulty yields r
        # vetoes, i.e. r further independent draws.
        vt = int(stick[x])
        r_veto = 0
        while r_veto < n and byz[(ev + r_veto) % n]:
            r_veto += 1
        veto_n += r_veto > 0
        for k, nm in ((1, "ChronoSeq"), (4, "ChronoSeq-k4"), (16, "ChronoSeq-k16")):
            win = min(int(v) for v in rng.integers(0, 2**63, k, dtype=np.int64)) * 2 < vt
            for _ in range(r_veto):
                if win:
                    break
                vt = int(rng.integers(0, 2**63, dtype=np.int64)) * 2
                win = min(int(v) for v in rng.integers(0, 2**63, k, dtype=np.int64)) * 2 < vt
            vt = int(stick[x])
            res[nm] = earlier or (same and win)
        res["Seed-before-cut"] = e_adv <= ev_eff
        # rotating-leader BFT: faulty leader reorders; honest leader orders by
        # arrival, so the injected tx must win the race to that leader
        el = int(lr.tx_epoch[x])
        if el < 0:
            res["Leader-BFT"] = False
        else:
            p = int(lr.Ep.proposer[el])
            if byz[p]:
                res["Leader-BFT"] = True
            else:
                arr_adv = (t_obs + sample_delay(rng, D[B, p], cfg)).min()
                res["Leader-BFT"] = bool(arr_adv < L.avail[g, p] and arr_adv <= lr.Ep.P[el])
        for k in names:
            succ[k] += bool(res[k])
            succ_h[k] += bool(res[k])
    k_all = len(victims)
    rows = []
    for s in names:
        rows.append(dict(f=f_byz, seed=seed, system=s, success=succ[s] / k_all,
                         success_honest_ingress=succ_h[s] / max(1, n_h)))
    rows.append(dict(f=f_byz, seed=seed, system="FIFO (Byzantine leader)", success=1.0, success_honest_ingress=1.0))
    for s, v in (("_reach", reach), ("_exclusion", excl_n), ("_veto", veto_n)):
        rows.append(dict(f=f_byz, seed=seed, system=s, success=np.nan, success_honest_ingress=v / max(1, n_h)))
    rows.append(dict(f=f_byz, seed=seed, system="_byz_ingress", success=byz_in / k_all, success_honest_ingress=np.nan))
    rows.append(dict(f=f_byz, seed=seed, system="_lagmax", success=float(Ep.lags.max()), success_honest_ingress=float((Ep.views > 0).sum())))
    return rows


def vertex_race(run, ev, g_victim, k_adv, rng) -> bool:
    """Random-priority lane-chain merge of epoch ev plus k adversarial roots."""
    import heapq
    L, Ep = run.L, run.Ep
    prev = Ep.cut[ev - 1] if ev > 0 else np.zeros(run.cfg.n, dtype=np.int64)
    cur = Ep.cut[ev]
    heap = []
    chains = {}
    for i in range(run.cfg.n):
        if cur[i] > prev[i]:
            gs = list(range(L.lane_first[i] + prev[i], L.lane_first[i] + cur[i]))
            chains[i] = gs
            heap.append((rng.random(), i, 0))
    for a in range(k_adv):
        heap.append((rng.random(), -1 - a, 0))
    heapq.heapify(heap)
    while heap:
        tk, ln, idx = heapq.heappop(heap)
        if ln < 0:
            return True
        gg = chains[ln][idx]
        if gg == g_victim:
            return False
        if idx + 1 < len(chains[ln]):
            heapq.heappush(heap, (rng.random(), ln, idx + 1))
    return False


# ---------------------------------------------------------------------------
# E8  censorship, E9 withholding / silent Byzantine nodes
# ---------------------------------------------------------------------------

def censor_excluder(w, L):
    tv = np.zeros(len(L.lane), dtype=bool)
    tv[L.tx_vertex[(L.tx_vertex >= 0) & w.targeted]] = True
    per_lane = [np.flatnonzero(tv[L.lane_first[i]:L.lane_first[i] + L.lane_count[i]])
                for i in range(len(L.lane_first))]

    def fn(t, p, P, C, C_prev, floor_max):
        C = C.copy()
        for i, hs in enumerate(per_lane):
            k = np.searchsorted(hs, C_prev[i])
            if k < hs.size and hs[k] < C[i]:
                C[i] = max(int(floor_max[i]), int(hs[k]), int(C_prev[i]))
        return C
    return fn


def exp_censor(args):
    (f_byz,), seed = args
    cfg = Config(f_byz=f_byz, byz_censor=True, censor_frac=0.10, duration=15)
    run = simulate_dag(cfg, seed, exclude_factory=censor_excluder)
    m = steady(run, cfg)
    lat = run.latency(T_FAST)
    tg = m & run.w.targeted
    ut = m & ~run.w.targeted
    return dict(f=f_byz, seed=seed, tgt_p50=pct(lat[tg], 50), tgt_p99=pct(lat[tg], 99),
                tgt_max=float(np.max(lat[tg])) if tg.any() else np.nan,
                tgt_included=float(np.isfinite(lat[tg]).mean()),
                other_p50=pct(lat[ut], 50), other_p99=pct(lat[ut], 99),
                vc=int((run.Ep.views > 0).sum()), epochs=len(run.Ep.P), lagmax=float(run.Ep.lags.max()))


def exp_withhold(args):
    (f_byz,), seed = args
    cfg = Config(f_byz=f_byz, byz_withhold=True, byz_silent=True, duration=15)
    run = simulate_dag(cfg, seed)
    m = steady(run, cfg)
    lat = run.latency(T_FAST)[m]
    return dict(f=f_byz, seed=seed, p50=pct(lat, 50), p95=pct(lat, 95), p99=pct(lat, 99),
                vc_frac=float((run.Ep.views > 0).mean()),
                cut_interval=float(np.median(np.diff(run.Ep.P))))


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def _grid(name):
    if name == "consistency":
        return [(c,) for c in [(0.05, 0.0, 0), (0.25, 0.01, 0), (0.5, 0.05, 0),
                                (0.05, 0.0, 3), (0.25, 0.01, 3), (0.5, 0.05, 3)]]
    if name == "latency":
        return [((n,),) for n in (8, 16, 32, 64, 128)]
    if name == "load":
        return [((l,),) for l in LOADS]
    if name == "microblock":
        return [((k, l),) for l in (50000, 150000) for k in MB_KIB]
    if name == "fairness":
        return [((wl,),) for wl in ("uniform", "zipf", "defi")]
    if name == "race":
        return [(("copies",),), (("sybil",),)]
    if name in ("frontrun",):
        return [((f,),) for f in (1, 4, 7, 10)]
    if name in ("censor", "withhold"):
        return [((f,),) for f in (0, 1, 4, 7, 10)]
    raise ValueError(name)


FUNCS = dict(consistency=exp_consistency, latency=exp_latency, load=exp_load,
             microblock=exp_microblock, fairness=exp_fairness, race=exp_race,
             frontrun=exp_frontrun, censor=exp_censor, withhold=exp_withhold)


def _call(task):
    name, cfgkey, seed = task
    return name, FUNCS[name]((cfgkey[0], seed))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    names = a.only or list(FUNCS)
    for name in names:
        t0 = time.time()
        tasks = [(name, key, s) for key in _grid(name) for s in range(1, a.seeds + 1)]
        with Pool(a.procs) as pool:
            res = [r for _, r in pool.imap_unordered(_call, tasks)]
        write(name, res, a.out)
        print(f"{name}: {len(tasks)} runs in {time.time() - t0:.0f}s", flush=True)


def write(name, res, out):
    p = lambda f: os.path.join(out, f)
    if name == "latency":
        pd.DataFrame([r for x in res for r in x["rows"]]).to_csv(p("latency_runs.csv"), index=False)
        pd.DataFrame([r for x in res for r in x["sweep"]]).to_csv(p("vdf_sweep_runs.csv"), index=False)
        pd.DataFrame([x["breakdown"] for x in res if x["breakdown"]]).to_csv(p("latency_breakdown_runs.csv"), index=False)
    elif name == "fairness":
        pd.DataFrame([x["jain"] for x in res]).to_csv(p("jain_runs.csv"), index=False)
        pd.DataFrame([r for x in res for r in x["inv"]]).to_csv(p("inversion_runs.csv"), index=False)
    elif name in ("load", "race", "frontrun"):
        pd.DataFrame([r for x in res for r in x]).to_csv(p(f"{name}_runs.csv"), index=False)
    else:
        pd.DataFrame(res).to_csv(p(f"{name}_runs.csv"), index=False)


if __name__ == "__main__":
    main()
