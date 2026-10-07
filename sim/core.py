"""Core model: network, workload, lanes, cut agreement, FIFO baseline."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Network model
# ---------------------------------------------------------------------------

REGIONS = ("SYD", "SIN", "PDX")
# Median RTTs (ms) between AWS ap-southeast-2, ap-southeast-1 and us-west-2
# (cloudping.co, p50 over one week, 2026-09-29 to 2026-10-06, retrieved
# 2026-10-06; inter-region values are the mean of the two directions:
# SYD-SIN 174.480/174.308, SYD-PDX 148.010/141.136, SIN-PDX 180.213/173.989).
RTT_MS = np.array([
    [1.16, 174.39, 144.57],
    [174.39, 1.22, 177.10],
    [144.57, 177.10, 9.13],
])
ONE_WAY_S = RTT_MS / 2000.0

MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)


@dataclass
class Config:
    n: int = 32                    # sequencer nodes (= lanes)
    duration: float = 20.0         # seconds of client submissions
    warmup: float = 2.0            # discard txs submitted before this time
    load_tps: float = 20000.0      # offered load
    workload: str = "uniform"      # uniform | zipf | defi
    n_clients: int = 999
    tx_bytes: int = 512
    hdr_bytes: int = 512           # vertex header + certificate bytes
    bw_Bps: float = 125e6          # 1 Gbps up and down per node
    mb_bytes: int = 64 * 1024      # micro-block payload cap
    mb_timeout: float = 0.050      # micro-block seal timeout
    access_ms: float = 5.0         # client -> in-region ingress, one way
    jitter_sigma: float = 0.05     # log-normal multiplicative jitter
    jitter_add_ms: float = 1.0     # additive exponential jitter (mean)
    spike_prob: float = 0.0        # probability of a latency spike per message
    spike_ms: float = 50.0         # mean spike size
    cut_min_interval: float = 0.050
    floor_lag: float = 0.240       # delta_f in the floor rule (= 2*delta, Thm 2)
    prop_window: float = 0.480     # tau_acc (= 4*delta, Thm 2): a proposal is accepted only within this
                                   # time of r_j (receipt of f+1 ENDORSEs of t-1)
    delta_bound: float = 0.120     # delta: post-GST one-way delay bound; W = tau_acc + 3*delta
    vc_timeout: float = 0.850      # view-change timeout tau_vc (> tau_acc + 3*delta, Thm 2)
    t_vdf: float = 0.850           # honest VDF evaluation time (A_max = 1)
    endorse: bool = True           # ChronoSeq: E_t is q ENDORSE signatures sent after finalizing t-1,
                                   # r_j is the receipt of f+1 of them, and a view-v>=1 leader waits
                                   # until r_p + 2*delta. False: baselines (E_t = the COMMITs of t-1).
    order_cost: float = 1e-6       # per-tx ticketing + sorting cost (s)
    # FIFO baseline
    fifo_leader_region: int = 2    # PDX
    fifo_fanout: int = 3           # 0 = direct broadcast to all n-1 replicas
    fifo_batch: float = 0.050      # feed batch interval
    # Ablation: all transactions enter one lane
    single_lane: bool = False
    # Byzantine behaviour
    f_byz: int = 0
    byz_withhold: bool = False     # Byzantine proposers never propose
    byz_silent: bool = False       # Byzantine nodes do not ack or vote
    byz_censor: bool = False       # Byzantine ingress drops targeted txs,
                                   # Byzantine proposers exclude them if allowed
    censor_frac: float = 0.0       # fraction of clients that are targeted
    resubmit_timeout: float = 0.5
    drain: float = 6.0             # extra simulated time to flush

    @property
    def f(self) -> int:
        return (self.n - 1) // 3

    @property
    def q(self) -> int:
        return self.n - self.f     # 2f+1 when n = 3f+1


def node_regions(n: int) -> np.ndarray:
    return np.arange(n) % 3


def sample_delay(rng, base, cfg: Config, size=None):
    """One-way delay samples around `base` (seconds), one independent draw per
    element of `base` (or per element of `size` if given)."""
    if size is None:
        size = np.shape(base)
    d = base * np.exp(cfg.jitter_sigma * rng.standard_normal(size))
    d = d + rng.exponential(cfg.jitter_add_ms / 1000.0, size)
    if cfg.spike_prob > 0:
        spikes = rng.random(size) < cfg.spike_prob
        d = d + spikes * rng.exponential(cfg.spike_ms / 1000.0, size)
    return d


def lindley(arrivals: np.ndarray, service: np.ndarray) -> np.ndarray:
    """Completion times of a FIFO single server (arrivals sorted)."""
    if len(arrivals) == 0:
        return arrivals.copy()
    c = np.cumsum(service)
    return c + np.maximum.accumulate(arrivals - (c - service))


def mix64(x: np.ndarray) -> np.ndarray:
    """splitmix64 finaliser; stands in for the random-oracle hash H."""
    with np.errstate(over="ignore"):
        z = (x.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15))
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def h64(*parts) -> int:
    h = hashlib.sha256()
    for p in parts:
        h.update(repr(p).encode())
    return int.from_bytes(h.digest()[:8], "big")


# ---------------------------------------------------------------------------
# Workload
# ---------------------------------------------------------------------------

@dataclass
class Workload:
    submit: np.ndarray      # submission time
    client: np.ndarray      # client id (= sender account)
    region: np.ndarray      # client region
    nonce: np.ndarray       # per-sender nonce
    txhash: np.ndarray      # uint64 identifier used for tickets / dedup
    tag: np.ndarray         # 0 normal, >0 special (race events)
    targeted: np.ndarray    # bool: client targeted by censors
    ingress_hint: np.ndarray  # -1 = home node, else forced ingress node

    def __len__(self):
        return len(self.submit)


def gen_workload(cfg: Config, rng) -> Workload:
    U, T = cfg.n_clients, cfg.duration
    if cfg.workload == "defi":
        burst = 20
        nb = rng.poisson(cfg.load_tps * T / burst)
        bt = rng.uniform(0, T, nb)
        t = (bt[:, None] + rng.uniform(0, 0.02, (nb, burst))).ravel()
        top = max(1, U // 20)
        hot = rng.random(t.size) < 0.8
        c = np.where(hot, rng.integers(0, top, t.size), rng.integers(0, U, t.size))
    else:
        m = rng.poisson(cfg.load_tps * T)
        t = rng.uniform(0, T, m)
        if cfg.workload == "zipf":
            p = 1.0 / np.arange(1, U + 1) ** 1.2
            p /= p.sum()
            perm = rng.permutation(U)
            c = perm[rng.choice(U, m, p=p)]
        else:
            c = rng.integers(0, U, m)
    order = np.argsort(t, kind="stable")
    t, c = t[order], c[order]
    m = t.size
    # per-sender nonces in submission order
    idx = np.lexsort((np.arange(m), c))
    cs = c[idx]
    first = np.r_[True, cs[1:] != cs[:-1]]
    grp_start = np.maximum.accumulate(np.where(first, np.arange(m), 0))
    nonce = np.empty(m, dtype=np.int64)
    nonce[idx] = np.arange(m) - grp_start
    targeted_clients = rng.random(U) < cfg.censor_frac
    return Workload(
        submit=t, client=c, region=c % 3, nonce=nonce,
        txhash=rng.integers(0, 2**63 - 1, m, dtype=np.int64).astype(np.uint64),
        tag=np.zeros(m, dtype=np.int64),
        targeted=targeted_clients[c],
        ingress_hint=np.full(m, -1, dtype=np.int64),
    )


def append_txs(w: Workload, **cols) -> Workload:
    k = len(cols["submit"])
    defaults = dict(
        nonce=np.zeros(k, dtype=np.int64),
        tag=np.zeros(k, dtype=np.int64),
        targeted=np.zeros(k, dtype=bool),
        ingress_hint=np.full(k, -1, dtype=np.int64),
    )
    defaults.update(cols)
    new = {}
    for name in ("submit", "client", "region", "nonce", "txhash", "tag",
                 "targeted", "ingress_hint"):
        new[name] = np.concatenate([getattr(w, name), np.asarray(defaults[name]).astype(getattr(w, name).dtype)])
    order = np.argsort(new["submit"], kind="stable")
    return Workload(**{k2: v[order] for k2, v in new.items()})


# ---------------------------------------------------------------------------
# Ingress assignment (incl. censorship with client resubmission)
# ---------------------------------------------------------------------------

def assign_ingress(cfg: Config, w: Workload, byz: np.ndarray, rng):
    """Returns (ingress node, arrival time at ingress)."""
    n = cfg.n
    nreg = node_regions(n)
    m = len(w)
    region_nodes = [np.flatnonzero(nreg == r) for r in range(3)]
    if cfg.single_lane:
        home = np.full(m, int(np.flatnonzero(nreg == cfg.fifo_leader_region)[0]))
    else:
        # each client has a fixed home node inside its region
        rn = np.array([len(region_nodes[r]) for r in range(3)])
        hc = mix64(w.client.astype(np.uint64) + np.uint64(12345)) % rn[w.region].astype(np.uint64)
        home = np.array([region_nodes[r][k] for r, k in zip(w.region, hc.astype(np.int64))], dtype=np.int64)
    home = np.where(w.ingress_hint >= 0, w.ingress_hint, home)
    ingress = home.copy()
    extra = np.zeros(m)
    if cfg.byz_censor and byz.any():
        bad = w.targeted & byz[ingress]
        for k in np.flatnonzero(bad):
            # client times out and resubmits to other nodes (own region first)
            r = w.region[k]
            cand = list(rng.permutation(region_nodes[r])) + list(rng.permutation(np.flatnonzero(nreg != r)))
            tries = 0
            for node in cand:
                if node == home[k]:
                    continue
                tries += 1
                if not byz[node]:
                    ingress[k] = node
                    break
            extra[k] = tries * cfg.resubmit_timeout
    base = ONE_WAY_S[w.region, nreg[ingress]] + cfg.access_ms / 1000.0
    arr = w.submit + extra + sample_delay(rng, base, cfg, m)
    return ingress, arr


# ---------------------------------------------------------------------------
# Lanes: micro-blocks, dissemination, availability certificates
# ---------------------------------------------------------------------------

@dataclass
class Lanes:
    lane: np.ndarray        # per vertex
    height: np.ndarray      # 0-based height in lane
    created: np.ndarray
    ntx: np.ndarray
    tx_start: np.ndarray    # into tx_order
    tx_end: np.ndarray
    tx_order: np.ndarray    # tx indices, lane by lane, vertex by vertex
    tx_vertex: np.ndarray   # per tx: vertex gid (-1 if never included)
    tx_pos: np.ndarray      # per tx: position inside its vertex
    avail: np.ndarray       # V x n : time vertex data is available at node
    cert: np.ndarray        # V : certificate formed at creator
    cert_arr: np.ndarray    # V x n : certificate known at node
    lane_first: np.ndarray  # first gid of each lane
    lane_count: np.ndarray  # vertices per lane
    sufmin: List[np.ndarray] = field(default_factory=list)   # per lane (V_i x n)
    premax: List[np.ndarray] = field(default_factory=list)   # per lane (V_i x n)

    def tips(self, lane: int, node: int, t: float) -> int:
        sm = self.sufmin[lane]
        if sm.shape[0] == 0:
            return 0
        return int(np.searchsorted(sm[:, node], t, side="right"))

    def tips_all(self, node: int, t: float) -> np.ndarray:
        return np.array([self.tips(i, node, t) for i in range(len(self.sufmin))], dtype=np.int64)


def form_microblocks(arr: np.ndarray, cfg: Config):
    K = max(1, cfg.mb_bytes // cfg.tx_bytes)
    starts, ends, seal = [], [], []
    s, m = 0, len(arr)
    while s < m:
        e_time = int(np.searchsorted(arr, arr[s] + cfg.mb_timeout, side="right"))
        if s + K <= e_time:
            e = s + K
            st = arr[e - 1]
        else:
            e = e_time
            st = arr[s] + cfg.mb_timeout
        starts.append(s)
        ends.append(e)
        seal.append(st)
        s = e
    return np.array(starts, dtype=np.int64), np.array(ends, dtype=np.int64), np.array(seal)


def build_lanes(cfg: Config, w: Workload, ingress: np.ndarray, arr: np.ndarray,
                byz: np.ndarray, rng, drop: Optional[np.ndarray] = None) -> Lanes:
    n, B = cfg.n, cfg.bw_Bps
    nreg = node_regions(n)
    D = ONE_WAY_S[nreg[:, None], nreg[None, :]]
    m = len(w)
    keep = np.ones(m, dtype=bool) if drop is None else ~drop
    lanes_v = []
    tx_order_parts = []
    tx_vertex = np.full(m, -1, dtype=np.int64)
    tx_pos = np.full(m, -1, dtype=np.int64)
    gid = 0
    V_lane, V_h, V_created, V_ntx, V_s, V_e = [], [], [], [], [], []
    offset = 0
    lane_first = np.zeros(n, dtype=np.int64)
    lane_count = np.zeros(n, dtype=np.int64)
    for i in range(n):
        idx = np.flatnonzero((ingress == i) & keep)
        idx = idx[np.argsort(arr[idx], kind="stable")]
        a = arr[idx]
        s, e, seal = form_microblocks(a, cfg)
        lane_first[i] = gid
        lane_count[i] = len(s)
        for h in range(len(s)):
            tx_vertex[idx[s[h]:e[h]]] = gid + h
            tx_pos[idx[s[h]:e[h]]] = np.arange(e[h] - s[h])
        V_lane.append(np.full(len(s), i))
        V_h.append(np.arange(len(s)))
        V_created.append(seal)
        V_ntx.append(e - s)
        V_s.append(offset + s)
        V_e.append(offset + e)
        tx_order_parts.append(idx)
        offset += len(idx)
        gid += len(s)
    V = gid
    lane = np.concatenate(V_lane).astype(np.int64) if V else np.zeros(0, dtype=np.int64)
    height = np.concatenate(V_h).astype(np.int64) if V else np.zeros(0, dtype=np.int64)
    created = np.concatenate(V_created) if V else np.zeros(0)
    ntx = np.concatenate(V_ntx).astype(np.int64) if V else np.zeros(0, dtype=np.int64)
    size = ntx * cfg.tx_bytes + cfg.hdr_bytes
    avail = np.full((V, n), np.inf)
    # --- uplink: direct broadcast to n-1 peers, random send order ---
    raw_by_node: List[List[np.ndarray]] = [[] for _ in range(n)]
    for i in range(n):
        g0, c = lane_first[i], lane_count[i]
        if c == 0:
            continue
        sl = slice(g0, g0 + c)
        occ = (n - 1) * size[sl] / B
        start = lindley(created[sl], occ) - occ
        ranks = rng.permuted(np.tile(np.arange(n - 1), (c, 1)), axis=1)
        comp = start[:, None] + (ranks + 1) * (size[sl] / B)[:, None]
        peers = np.array([j for j in range(n) if j != i])
        raw = comp + sample_delay(rng, D[i, peers][None, :], cfg, (c, n - 1))
        avail[sl, i] = created[sl]
        for k, j in enumerate(peers):
            raw_by_node[j].append(np.stack([raw[:, k], np.arange(g0, g0 + c, dtype=float)]))
    # --- downlink queues ---
    for j in range(n):
        if not raw_by_node[j]:
            continue
        R = np.concatenate(raw_by_node[j], axis=1)
        order = np.argsort(R[0], kind="stable")
        r, g = R[0][order], R[1][order].astype(np.int64)
        done = lindley(r, size[g] / B)
        avail[g, j] = done
    # --- acks and certificates (in-lane order enforced per receiver) ---
    cert = np.full(V, np.inf)
    cert_arr = np.full((V, n), np.inf)
    sufmin, premax = [], []
    q = cfg.q
    for i in range(n):
        g0, c = lane_first[i], lane_count[i]
        if c == 0:
            sufmin.append(np.zeros((0, n)))
            premax.append(np.zeros((0, n)))
            continue
        sl = slice(g0, g0 + c)
        pm = np.maximum.accumulate(avail[sl], axis=0)
        ack = pm.copy()
        if cfg.byz_silent:
            silent = byz.copy()
            silent[i] = False
            ack[:, silent] = np.inf
        ack_arr = ack + sample_delay(rng, D[:, i][None, :], cfg, (c, n))
        ack_arr[:, i] = created[sl]
        ct = np.partition(ack_arr, q - 1, axis=1)[:, q - 1]
        cert[sl] = ct
        ca = ct[:, None] + sample_delay(rng, D[i, :][None, :], cfg, (c, n))
        ca[:, i] = ct
        cert_arr[sl] = ca
        sufmin.append(np.minimum.accumulate(ca[::-1], axis=0)[::-1])
        premax.append(pm)
    tx_order = np.concatenate(tx_order_parts) if tx_order_parts else np.zeros(0, dtype=np.int64)
    return Lanes(lane=lane, height=height, created=created, ntx=ntx,
                 tx_start=np.concatenate(V_s) if V else np.zeros(0, dtype=np.int64),
                 tx_end=np.concatenate(V_e) if V else np.zeros(0, dtype=np.int64),
                 tx_order=tx_order, tx_vertex=tx_vertex, tx_pos=tx_pos,
                 avail=avail, cert=cert, cert_arr=cert_arr,
                 lane_first=lane_first, lane_count=lane_count,
                 sufmin=sufmin, premax=premax)


# ---------------------------------------------------------------------------
# Cut agreement (two-phase, PBFT-style, rotating proposer, floor rule)
# ---------------------------------------------------------------------------

@dataclass
class Epochs:
    cut: np.ndarray        # E x n  lane heights (count of included vertices)
    proposer: np.ndarray   # E
    views: np.ndarray      # E  number of failed views before commit
    P: np.ndarray          # E  proposal time
    rv: np.ndarray         # E x n proposal received (speculative VDF start)
    pc: np.ndarray         # E x n prepare certificate time
    cc: np.ndarray         # E x n commit certificate time
    data: np.ndarray       # E x n all vertex data of the cut available
    seed: np.ndarray       # E uint64
    floor_max: np.ndarray = None   # E x n  max honest floor (for adversaries)
    floor_sorted: np.ndarray = None  # E x honest x n, honest floors sorted per lane
    lags: np.ndarray = None        # honest proposal-arrival lags (view 0)
    w_emp: np.ndarray = None       # empirical seed-influence window per epoch
    ref_spread: np.ndarray = None  # max honest r_j - earliest adversary input time
    floor_rejects: int = 0
    lock_fails: int = 0            # view-0 failures caused by the local lock deadline
    timer_late: np.ndarray = None  # E  view-0 timer fired at f+1 honest nodes before finalization

    def seq_time(self, cfg: Config, t_vdf: Optional[float] = None, ntx: Optional[np.ndarray] = None):
        """Per-epoch, per-node time at which the order of cut t is final."""
        T = cfg.t_vdf if t_vdf is None else t_vdf
        base = np.maximum(self.cc, self.data)
        if T > 0:
            # nodes start the VDF speculatively when the proposal arrives
            base = np.maximum(base, self.rv + T)
        if ntx is not None:
            base = base + (ntx * cfg.order_cost)[:, None]
        return np.maximum.accumulate(base, axis=0)


def _new_view_time(arr: np.ndarray, cfg: Config) -> float:
    """Time the next leader proposes after a view change: it needs q VIEW-CHANGEs;
    with endorse=True it also waits until 2*delta after its reference time (f+1)
    so that every honest lock is reported (Algorithm 1)."""
    arr = np.sort(arr)
    t_q = arr[min(cfg.q, arr.size) - 1]
    if cfg.endorse:
        t_q = max(t_q, arr[min(cfg.f, arr.size - 1)] + 2 * cfg.delta_bound)
    return float(t_q)


def run_agreement(cfg: Config, L: Lanes, byz: np.ndarray, rng,
                  censor_heights: Optional[List[np.ndarray]] = None,
                  exclude_fn=None) -> Epochs:
    """Sequential epochs; each epoch commits one monotone cut."""
    n, q = cfg.n, cfg.q
    nreg = node_regions(n)
    D = ONE_WAY_S[nreg[:, None], nreg[None, :]]
    honest = ~byz
    voters = honest if cfg.byz_silent else np.ones(n, dtype=bool)
    total = L.lane_count.copy()
    C_prev = np.zeros(n, dtype=np.int64)
    commit_prev = np.zeros(n)
    last_P = -np.inf
    t = 0
    cuts, props, views, Ps, rvs, pcs, ccs, datas, seeds = [], [], [], [], [], [], [], [], []
    rejects = 0
    lock_fails = 0
    timer_late = []
    lags, floor_max, wemp, floor_sorted, spread = [], [], [], [], []
    ref = np.zeros(n)          # r_j: receipt of f+1 ENDORSEs (baselines: COMMITs) of epoch t-1
    horizon = cfg.duration + cfg.drain
    while True:
        start = commit_prev.copy()
        v = 0
        view_start = ref.copy() if cfg.endorse else start.copy()
        earliest = None
        while True:
            p = (t + v) % n
            if v == 0:
                E = max(start[p], last_P + cfg.cut_min_interval)
            else:
                E = earliest
            if byz[p] and cfg.byz_withhold:
                # honest nodes time out and send VIEW-CHANGE to next proposer
                vc_send = view_start + cfg.vc_timeout
                p2 = (t + v + 1) % n
                earliest = _new_view_time(vc_send[honest] + sample_delay(rng, D[honest, p2], cfg), cfg)
                view_start = vc_send
                v += 1
                continue
            P = E
            C = np.maximum(C_prev, L.tips_all(p, P))
            # floor rule: every certificate a voter held floor_lag before its
            # epoch start must be included
            floors = np.array([L.tips_all(j, ref[j] - cfg.floor_lag) for j in range(n)])
            fs = np.sort(floors[honest], axis=0)
            if byz[p] and exclude_fn is not None:
                # a faulty leader only needs q - f_byz honest acceptors
                need = max(1, q - (int(byz.sum()) if not cfg.byz_silent else 0))
                # a rational censor picks the `need` honest nodes with the earliest
                # reference times and excludes only what all of them accept
                hidx = np.flatnonzero(honest)
                S = hidx[np.argsort(ref[hidx], kind="stable")[:min(need, hidx.size)]]
                C = exclude_fn(t, p, P, C, C_prev, floors[S].max(axis=0))
            recv = P + sample_delay(rng, D[p, :], cfg)
            recv[p] = P
            valid = (C[None, :] >= floors).all(axis=1)
            if v == 0:
                lag = recv - ref
                lags.append(lag[honest])
                valid &= lag <= cfg.prop_window
            valid |= byz
            n_ok = int((valid & voters).sum())
            ok = n_ok >= q
            if ok:
                vote = valid & voters
                # prepare phase
                vidx = np.flatnonzero(vote)
                rows = np.arange(vidx.size)
                pv = recv[vidx][:, None] + sample_delay(rng, D[vidx, :], cfg)
                pv[rows, vidx] = recv[vidx]          # own vote needs no network hop
                pc = np.partition(pv, q - 1, axis=0)[q - 1]
                pc = np.maximum(pc, recv)  # a node needs the proposal itself
                if v == 0:
                    # local lock deadline (Alg. 1): lock and send COMMIT only if q
                    # PREPAREs arrive by r_j + tau_acc + 2*delta; faulty voters ignore it
                    lock = vote & ((pc <= ref + cfg.prop_window + 2 * cfg.delta_bound) | byz)
                    if int(lock.sum()) < q:
                        lock_fails += 1
                        ok = False
                else:
                    lock = vote
            if not ok:
                rejects += 1
                vc_send = view_start + cfg.vc_timeout
                p2 = (t + v + 1) % n
                earliest = _new_view_time(vc_send[honest] + sample_delay(rng, D[honest, p2], cfg), cfg)
                view_start = vc_send
                v += 1
                continue
            break
        floor_max.append(floors[honest].max(axis=0))
        floor_sorted.append(fs)
        # commit phase (locked voters only)
        lidx = np.flatnonzero(lock)
        lrows = np.arange(lidx.size)
        cv = pc[lidx][:, None] + sample_delay(rng, D[lidx, :], cfg)
        cv[lrows, lidx] = pc[lidx]
        cc = np.partition(cv, q - 1, axis=0)[q - 1]
        cc = np.maximum(cc, pc)
        # diagnostic: the view-0 timer (tau_vc after the view start) expires at some
        # honest node before it finalizes; harmless for the outcome because every
        # COMMIT is sent before the lock deadline < tau_vc, but counted
        timer_late.append(bool((cc - view_start)[honest].max() > cfg.vc_timeout))
        # data availability of the whole cut at each node
        data = np.zeros(n)
        for i in range(n):
            if C[i] > 0:
                data = np.maximum(data, L.premax[i][C[i] - 1])
        # empirical seed window: from the earliest time an adversary can hold
        # f+1 honest inputs of E_t (ENDORSEs; baselines: COMMITs) to the last honest
        # acceptance deadline for epoch t (incl. one echo hop)
        if t > 0:
            # inputs of E_t: ENDORSEs (sent at finalization) or COMMITs (sent at lock)
            hp = np.sort((ccs[-1] if cfg.endorse else pcs[-1])[honest])
            a_t = hp[min(cfg.f, hp.size - 1)]
            # W_emp = (max honest r_j - a) + tau_acc + 2*delta (lock deadline)
            spread.append(ref[honest].max() - a_t)
            wemp.append(spread[-1] + cfg.prop_window + 2 * cfg.delta_bound)
        seed = h64("cut", t, C.tobytes())
        cuts.append(C.copy()); props.append(p); views.append(v); Ps.append(P)
        rvs.append(recv); pcs.append(pc); ccs.append(cc); datas.append(data); seeds.append(seed)
        C_prev = C
        if cfg.endorse:
            # every node that finalized broadcasts ENDORSE(t); the next leader builds
            # E_{t+1} from q of them, and r_j is the receipt of f+1 of them
            eidx = np.flatnonzero(voters)
            ev_ = cc[eidx][:, None] + sample_delay(rng, D[eidx, :], cfg)
            ev_[np.arange(eidx.size), eidx] = cc[eidx]
            commit_prev = np.maximum(np.partition(ev_, q - 1, axis=0)[q - 1], cc)
            ref = np.partition(ev_, cfg.f, axis=0)[cfg.f]
        else:
            commit_prev = cc
            # reference time for the next epoch: f+1-th COMMIT arrival at each node
            ref = np.maximum(np.partition(cv, min(cfg.f, cv.shape[0] - 1), axis=0)[min(cfg.f, cv.shape[0] - 1)], 0)
        last_P = P
        t += 1
        if (C_prev >= total).all() and P > cfg.duration:
            break
        if P > horizon + 30:
            break
    return Epochs(cut=np.array(cuts), proposer=np.array(props), views=np.array(views),
                  P=np.array(Ps), rv=np.array(rvs), pc=np.array(pcs), cc=np.array(ccs), data=np.array(datas),
                  seed=np.array(seeds, dtype=np.uint64), floor_max=np.array(floor_max),
                  floor_sorted=np.array(floor_sorted),
                  lags=np.concatenate(lags) if lags else np.zeros(0),
                  w_emp=np.array(wemp), ref_spread=np.array(spread), floor_rejects=rejects,
                  lock_fails=lock_fails, timer_late=np.array(timer_late))


def tx_epochs(L: Lanes, Ep: Epochs) -> np.ndarray:
    """Epoch index of every vertex (first cut whose lane height exceeds it)."""
    V = len(L.lane)
    e = np.full(V, -1, dtype=np.int64)
    for i in range(Ep.cut.shape[1]):
        g0, c = L.lane_first[i], L.lane_count[i]
        if c == 0:
            continue
        e[g0:g0 + c] = np.searchsorted(Ep.cut[:, i], np.arange(c), side="right")
    e[e >= Ep.cut.shape[0]] = -1
    return e


# ---------------------------------------------------------------------------
# Single-leader FIFO baseline
# ---------------------------------------------------------------------------

@dataclass
class FifoResult:
    order_time: np.ndarray   # arrival (ordering) time at the leader
    seq: np.ndarray          # time the batch leaves the leader's uplink
    included: np.ndarray


def run_fifo(cfg: Config, w: Workload, byz_leader: bool, rng) -> FifoResult:
    B = cfg.bw_Bps
    m = len(w)
    base = ONE_WAY_S[w.region, cfg.fifo_leader_region] + cfg.access_ms / 1000.0
    raw = w.submit + sample_delay(rng, base, cfg, m)
    included = np.ones(m, dtype=bool)
    if byz_leader and cfg.byz_censor:
        included &= ~w.targeted
    order = np.argsort(raw, kind="stable")
    done = np.full(m, np.inf)
    done[order] = lindley(raw[order], np.full(m, cfg.tx_bytes / B))
    bid = np.floor(done / cfg.fifo_batch).astype(np.int64)
    ub, inv, cnt = np.unique(bid, return_inverse=True, return_counts=True)
    cut_time = (ub + 1) * cfg.fifo_batch
    copies = (cfg.n - 1) if cfg.fifo_fanout == 0 else min(cfg.fifo_fanout, cfg.n - 1)
    occ = copies * (cnt * cfg.tx_bytes + cfg.hdr_bytes) / B
    finish = lindley(cut_time, occ)
    seq = finish[inv]
    seq = np.where(included, seq, np.inf)
    return FifoResult(order_time=done, seq=seq, included=included)


# ---------------------------------------------------------------------------
# Byzantine set
# ---------------------------------------------------------------------------

def pick_byz(cfg: Config, rng) -> np.ndarray:
    byz = np.zeros(cfg.n, dtype=bool)
    if cfg.f_byz > 0:
        byz[rng.choice(cfg.n, cfg.f_byz, replace=False)] = True
    return byz
