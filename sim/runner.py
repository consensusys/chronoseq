"""Run wrappers, ordering policies and metrics."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .core import (Config, Epochs, Lanes, Workload, append_txs, assign_ingress,
                   build_lanes, gen_workload, mix64, pick_byz, run_agreement,
                   run_fifo, tx_epochs, ONE_WAY_S, node_regions, sample_delay)


@dataclass
class DagRun:
    cfg: Config
    w: Workload
    byz: np.ndarray
    ingress: np.ndarray
    arr: np.ndarray
    L: Lanes
    Ep: Epochs
    v_epoch: np.ndarray     # per vertex
    tx_epoch: np.ndarray    # per tx (-1 if not included)
    ntx_epoch: np.ndarray

    def seq(self, t_vdf: float) -> np.ndarray:
        """Per-tx sequencing time at the ingress node (inf if never)."""
        st = self.Ep.seq_time(self.cfg, t_vdf=t_vdf, ntx=self.ntx_epoch)
        out = np.full(len(self.w), np.inf)
        ok = self.tx_epoch >= 0
        out[ok] = st[self.tx_epoch[ok], self.ingress[ok]]
        return out

    def latency(self, t_vdf: float) -> np.ndarray:
        return self.seq(t_vdf) - self.w.submit


def simulate_dag(cfg: Config, seed: int, w: Optional[Workload] = None,
                 workload_hook=None, exclude_factory=None) -> DagRun:
    rng = np.random.default_rng(seed)
    if w is None:
        w = gen_workload(cfg, rng)
    byz = pick_byz(cfg, rng)
    if workload_hook is not None:
        w = workload_hook(w, byz, rng)
    ingress, arr = assign_ingress(cfg, w, byz, rng)
    L = build_lanes(cfg, w, ingress, arr, byz, rng)
    exclude_fn = exclude_factory(w, L) if exclude_factory is not None else None
    Ep = run_agreement(cfg, L, byz, rng, exclude_fn=exclude_fn)
    ve = tx_epochs(L, Ep)
    te = np.where(L.tx_vertex >= 0, ve[np.maximum(L.tx_vertex, 0)], -1)
    ntx = np.bincount(te[te >= 0], minlength=len(Ep.P)).astype(float)
    return DagRun(cfg, w, byz, ingress, arr, L, Ep, ve, te, ntx)


# ---------------------------------------------------------------------------
# Ordering policies (return a sortable key per tx; lower = earlier)
# ---------------------------------------------------------------------------

def _rank(*keys) -> np.ndarray:
    """Global rank from lexicographic keys (last key most significant)."""
    order = np.lexsort(keys)
    r = np.empty(len(order), dtype=np.int64)
    r[order] = np.arange(len(order))
    return r


def sender_tickets(run: DagRun) -> np.ndarray:
    """Per-sender ticket of every tx: H(s_t || sender) (ChronoSeq, Alg. 2)."""
    w, e = run.w, run.tx_epoch
    ok = e >= 0
    seed = np.zeros(len(w), dtype=np.uint64)
    seed[ok] = run.Ep.seed[e[ok]]
    return mix64(w.client.astype(np.uint64) ^ mix64(seed))


def median_ts(run: DagRun) -> np.ndarray:
    """Pompe-style median receive timestamp of each tx (its micro-block)."""
    v = np.maximum(run.L.tx_vertex, 0)
    return np.median(run.L.avail, axis=1)[v]


def order_rank(run: DagRun, policy: str) -> np.ndarray:
    """Global position of every included tx under a DAG ordering policy.

    policy: 'cs'        ChronoSeq: per-sender tickets, nonce order inside a sender
            'cs_vertex' per-vertex tickets with lane-chain merge (ablation)
            'det'       deterministic Bullshark-style order (height, rotated lane)
            'median'    Pompe-style median receive timestamp inside the cut
    Copies of the same tx (same txhash) are kept; callers dedup by min rank.
    """
    L, Ep, w = run.L, run.Ep, run.w
    e = run.tx_epoch.copy()
    big = np.iinfo(np.int64).max // 4
    e_key = np.where(e >= 0, e, big)
    v = np.maximum(L.tx_vertex, 0)
    lane = L.lane[v]
    height = L.height[v]
    pos = np.maximum(L.tx_pos, 0)
    n = run.cfg.n
    if policy == "cs":
        return _rank(w.txhash, w.nonce, sender_tickets(run), e_key)
    if policy == "median":
        return _rank(w.txhash, median_ts(run), e_key)
    if policy == "det":
        prev = np.zeros(len(w), dtype=np.int64)
        ok = e > 0
        prev[ok] = Ep.cut[e[ok] - 1, lane[ok]]
        hoff = height - prev
        rot = (lane - np.where(e >= 0, e, 0)) % n
        return _rank(pos, rot, hoff, e_key)
    if policy == "cs_vertex":
        vrank = vertex_merge_rank(run)
        return _rank(pos, vrank[v], e_key)
    raise ValueError(policy)


def vertex_merge_rank(run: DagRun, extra=None) -> np.ndarray:
    """Per-vertex position within its epoch: random-priority Kahn merge over
    lane chains, tickets derived from the epoch seed (original design)."""
    L, Ep = run.L, run.Ep
    V = len(L.lane)
    tick = mix64(np.arange(V, dtype=np.uint64) * np.uint64(0x9E37) ^
                 mix64(Ep.seed[np.maximum(run.v_epoch, 0)]))
    rank = np.full(V, -1, dtype=np.int64)
    order = np.lexsort((L.height, L.lane, run.v_epoch))
    ve = run.v_epoch[order]
    bounds = np.flatnonzero(np.r_[True, ve[1:] != ve[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        if ve[a] < 0:
            continue
        seg = order[a:b]
        chains: Dict[int, List[int]] = {}
        for g in seg:
            chains.setdefault(int(L.lane[g]), []).append(int(g))
        heap = [(int(tick[c[0]]), ln, 0) for ln, c in chains.items()]
        heapq.heapify(heap)
        k = 0
        while heap:
            _, ln, i = heapq.heappop(heap)
            g = chains[ln][i]
            rank[g] = k
            k += 1
            if i + 1 < len(chains[ln]):
                heapq.heappush(heap, (int(tick[chains[ln][i + 1]]), ln, i + 1))
    return rank


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def ci95(x) -> tuple:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (np.nan, np.nan)
    m = x.mean()
    if x.size < 2:
        return (m, 0.0)
    from scipy import stats
    h = stats.t.ppf(0.975, x.size - 1) * x.std(ddof=1) / np.sqrt(x.size)
    return (m, h)


def jain(latency: np.ndarray, client: np.ndarray) -> float:
    ok = np.isfinite(latency)
    s = np.bincount(client[ok], weights=latency[ok])
    c = np.bincount(client[ok])
    mu = s[c > 0] / c[c > 0]
    return float(mu.sum() ** 2 / (mu.size * (mu ** 2).sum()))


GAP_BINS_MS = [0, 5, 10, 25, 50, 100, 200, 400, 800, 1600]


def inversion_by_gap(submit: np.ndarray, rank: np.ndarray, valid: np.ndarray,
                     rng, samples: int = 200000) -> np.ndarray:
    """Fraction of pairs (a, b) with submit(b) - submit(a) in each gap bin whose
    committed order is inverted (b before a)."""
    idx = np.flatnonzero(valid)
    s = submit[idx]
    r = rank[idx]
    o = np.argsort(s, kind="stable")
    s, r = s[o], r[o]
    out = []
    for lo, hi in zip(GAP_BINS_MS[:-1], GAP_BINS_MS[1:]):
        a = rng.integers(0, s.size, samples)
        g = rng.uniform(lo, hi, samples) / 1000.0
        b = np.searchsorted(s, s[a] + g)
        keep = b < s.size
        a, b = a[keep], b[keep]
        keep = s[b] > s[a]
        a, b = a[keep], b[keep]
        out.append(float((r[b] < r[a]).mean()) if a.size else np.nan)
    return np.array(out)


# ---------------------------------------------------------------------------
# Rotating-leader BFT sequencer baseline (HotStuff/HotShot-style committee):
# transactions are pre-disseminated by the same micro-block broadcast, the
# leader of each epoch proposes (by hash) everything it has received, orders it
# by its local arrival, and the committee commits with the same PBFT timing.
# ---------------------------------------------------------------------------

@dataclass
class LeaderRun:
    Ep: Epochs
    v_epoch: np.ndarray
    tx_epoch: np.ndarray
    ntx_epoch: np.ndarray


def simulate_leader_bft(run: DagRun, seed: int) -> LeaderRun:
    from copy import copy
    from dataclasses import replace as dreplace
    L = copy(run.L)
    L.sufmin, L.premax = [], []
    for i in range(run.cfg.n):
        g0, c = L.lane_first[i], L.lane_count[i]
        A = run.L.avail[g0:g0 + c]
        if c == 0:
            L.sufmin.append(np.zeros((0, run.cfg.n))); L.premax.append(np.zeros((0, run.cfg.n)))
            continue
        L.sufmin.append(np.minimum.accumulate(A[::-1], axis=0)[::-1])
        L.premax.append(np.maximum.accumulate(A, axis=0))
    cfg = dreplace(run.cfg, floor_lag=1e6, prop_window=1e6, endorse=False)
    Ep = run_agreement(cfg, L, run.byz, np.random.default_rng(seed + 4242))
    ve = tx_epochs(L, Ep)
    te = np.where(L.tx_vertex >= 0, ve[np.maximum(L.tx_vertex, 0)], -1)
    ntx = np.bincount(te[te >= 0], minlength=len(Ep.P)).astype(float)
    return LeaderRun(Ep, ve, te, ntx)


def leader_order_rank(run: DagRun, lr: LeaderRun) -> np.ndarray:
    """Leader's local arrival order inside each epoch."""
    L = run.L
    e = lr.tx_epoch
    big = np.iinfo(np.int64).max // 4
    v = np.maximum(L.tx_vertex, 0)
    lead = lr.Ep.proposer[np.maximum(e, 0)]
    arr_at_leader = L.avail[v, lead]
    return _rank(np.maximum(L.tx_pos, 0), arr_at_leader, np.where(e >= 0, e, big))
