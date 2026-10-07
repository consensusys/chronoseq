# ChronoSeq

**A Decentralized Layer-2 Sequencer with DAG Mempool and VDF Ordering**

> **Reproducing the paper.** The evaluation uses the timing simulator in
> [`sim/`](sim/) and the microbenchmarks in [`bench/`](bench/), not the
> single-process `benchmarks/` harness further below. The `chronoseq/` package
> and `benchmarks/` are a reference implementation of the data structures of
> an early design; they run in one process without a network and are not used
> in the paper.
>
> ```bash
> pip install numpy pandas scipy
> python -m sim.experiments --out results --seeds 10 --procs 2   # ~6 min on 2 cores
> python -m sim.aggregate   --inp results --out results/agg      # CSVs for figures
> python -m unittest tests.test_sim -v
>
> # CPU microbenchmarks (Section VI-E of the paper); needs gcc and OpenSSL headers
> pip install chiavdf==1.1.14 blspy==2.0.3 coincurve==21.0.0
> bash bench/run_all.sh                                          # ~3 min
>
> python -m sim.paper_tables --inp results --out <paper-dir>     # tables/*.tex + numbers.tex
> ```
>
> `results/` contains the raw per-seed CSVs and benchmark outputs used in the
> paper. Every number in the paper's text and tables is generated from them by
> `sim/paper_tables.py`. See the module docstring in `sim/__init__.py` for what
> is and is not modeled. Benchmarks were run on one 2.1 GHz Intel Xeon vCPU
> (2 vCPUs, Python 3.11, OpenSSL 3.0.13).

---

## Overview

ChronoSeq is a decentralized Layer-2 sequencer that combines a **DAG-based mempool** for parallel transaction dissemination with a **Verifiable Delay Function (VDF)-driven ordering mechanism** for fairness. It eliminates single-leader bottlenecks present in centralized rollup designs (e.g., Arbitrum, Optimism) while providing:

- **High throughput** via multi-lane parallel dissemination
- **Order-fairness** via VDF-seeded cryptographic ticket assignment
- **MEV resistance** via unpredictable epoch seeds
- **Multi-chain interoperability** via modular L1 settlement adapters

## Architecture

```
Clients ──→ [ DAG Mempool ] ──→ [ VDF Beacon ] ──→ [ Ticket Assignment ] ──→ [ DAG Merge ] ──→ [ Batch/Commit ]
                │                     │                     │                      │                    │
           Multi-lane            Epoch seed s_t       Anti-grinding           Causality-          L1 Adapters
           parallel gossip       (sequential          salt + lane             respecting          ETH / BTC /
           dissemination          squaring)            stagger                + aging              SOL
```

## Quick Start

### Prerequisites

- Python 3.10+
- No external dependencies for the core library (stdlib only)
- Optional: `numpy`, `matplotlib`, `pandas` for benchmarks and figures

### Installation

```bash
git clone https://github.com/andrewdong14/chronoseq.git
cd chronoseq

# Install in editable mode (core library — zero dependencies)
pip install -e .

# Install with benchmark dependencies
pip install -e ".[bench]"

# Install everything (bench + dev)
pip install -e ".[all]"
```

### Run Tests

```bash
# Using unittest (no extra dependencies)
python -m unittest discover -s tests -v

# Using pytest (if installed)
python -m pytest tests/ -v
```

### Run Benchmarks

```bash
# Quick mode (~10 seconds)
python -m benchmarks.run_all --quick --output-dir data/

# Full mode (~2-5 minutes)
python -m benchmarks.run_all --output-dir data/

# Custom random seed for reproducibility
python -m benchmarks.run_all --seed 42 --output-dir data/
```

### Generate Figures

```bash
# Requires: pip install matplotlib pandas numpy
python scripts/gen_figures.py --data-dir data/ --output-dir figures/
```

Produces PDF and PNG figures:

| File | Description |
|------|-------------|
| `fig_throughput_lanes.pdf` | Throughput vs. lane count |
| `fig_latency_percentiles.pdf` | Latency percentiles (p50 / p95 / p99) |
| `fig_fairness_jain.pdf` | Jain's fairness index across workloads |
| `fig_throughput_blocksize.pdf` | Throughput vs. micro-block size |

---

## Components

### DAG Mempool

The DAG mempool replaces the conventional linear queue with a directed acyclic graph of **transaction vertices** (micro-blocks). Each node maintains one **lane** — a sequence of vertices it produces.

Key classes:

- **`Transaction`** — a single client transaction with sender, payload, nonce, and SHA-256 hash
- **`Vertex`** — a micro-block containing transactions, Merkle commitment, creator ID, causality edges (parent references), and lane/sequence metadata
- **`DAGMempool`** — the multi-lane graph: manages vertex creation, parent referencing (intra-lane and cross-lane), induced subgraph extraction, and ready-set queries

```python
from chronoseq.dag import DAGMempool, Transaction

mempool = DAGMempool(num_lanes=16, max_txs_per_vertex=64)

tx = Transaction(sender="alice", payload=b"transfer 100 ETH to bob", nonce=1)
vertex = mempool.create_vertex(creator="node_0", lane=0, txs=[tx])

print(f"Vertex hash: {vertex.vertex_hash.hex()}")
print(f"Tx root:     {vertex.tx_root.hex()}")
print(f"Parents:     {[p.hex() for p in vertex.parents]}")
```

### VDF Epoch Beacon

At each epoch boundary, the beacon:

1. Evaluates a VDF (sequential squaring in ℤ/nℤ) on the previous seed — Θ(λ) sequential steps
2. Collects quorum endorsements
3. Derives the new epoch seed: `s_t = H(BEACON || t || y || qc)`
4. Retargets difficulty λ to keep VDF time in [0.5Δ, 0.8Δ]
5. Adjusts epoch length based on backlog

```python
from chronoseq.vdf import VDFBeacon

beacon = VDFBeacon(initial_difficulty=1024, epoch_length_ms=500)
meta = beacon.advance_epoch(endorsements=10, ready_vertices=500)

print(f"Epoch {meta.epoch_id}: seed={meta.seed.hex()[:16]}...")
print(f"VDF time: {meta.vdf_time_ms:.1f} ms, degraded: {meta.degraded}")
```

### Ticket Assignment & DAG Merge

**Ticket Assignment:**
- Anti-grinding salt: `salt[v] = H(SALT || v.hash || v.txRoot || meta[v])`
- Per-origin rate-cap weighting (penalises spamming origins)
- Ticket: `H(TICKET || s_t || salt[v])` with lexicographic hash tie-break
- Lane staggering: `Mix(ticket, H(LANE || s_t || lane_id))` prevents head-of-line blocking

**DAG Merge:**
- Kahn's topological sort with min-heap keyed by ticket value
- Causality-respecting: parents always ordered before children
- Light aging: `ticket[x] -= γ · k(x)` (γ = 2⁻¹⁶) prevents starvation
- Batch cutting by size (B_max) or elapsed time (T_max)

```python
from chronoseq.merge import assign_tickets, dag_merge

# After creating vertices and advancing the beacon:
tickets = assign_tickets(ready_set, epoch_seed)
vertices, edges = mempool.get_induced_subgraph(ready_set)
ordered, batches = dag_merge(ready_set, edges, tickets, batch_max=256)

print(f"Ordered {len(ordered)} vertices into {len(batches)} batches")
```

### L1 Settlement Adapters

Modular adapters translate ChronoSeq batch digests into L1-specific commitments:

| Adapter | L1 | Commitment Format |
|---------|----|--------------------|
| `EthereumAdapter` | Ethereum | Calldata (32-byte digest + batch ID) |
| `BitcoinAdapter` | Bitcoin | OP_RETURN (32-byte digest) |
| `SolanaAdapter` | Solana | Instruction packet |

```python
from chronoseq.adapters import EthereumAdapter

eth = EthereumAdapter()
receipt = eth.commit_batch(batch)
print(f"L1 tx: {receipt.l1_tx_hash.hex()}, gas: {receipt.gas_used}")
```

### Network Simulation

The `GossipNetwork` class simulates multi-node gossip dissemination for benchmarking:

- Configurable node count, fan-out, and latency parameters
- Each node has a dedicated lane and produces micro-blocks
- Cross-lane causality edges created with configurable probability
- Gossip latency modelled as O(log n)

The `SingleLeaderSequencer` provides a centralized FIFO baseline for comparison.

---

## Benchmarks

The benchmark harness measures ChronoSeq against a centralized single-leader baseline across four experiments:

| # | Experiment | Output CSV |
|---|------------|------------|
| 1 | Throughput vs. Lanes | `throughput_by_lanes.csv` |
| 2 | Latency Percentiles | `latency_percentiles.csv` |
| 3 | Fairness (Jain's Index) | `fairness_jain.csv` |
| 4 | Throughput vs. Block Size | `throughput_by_blocksize.csv` |
| 5 | Summary Comparison | `comparison_metrics.csv` |

### Configuration Parameters

| Parameter | Symbol | Code Variable | Default |
|-----------|--------|---------------|---------|
| VDF difficulty | λ | `initial_difficulty` | 64 (bench) / 2¹⁸ (production) |
| Epoch length | Δ | `epoch_length_ms` | 500 ms |
| Batch size | B_max | `batch_max` | 256 |
| Aging factor | γ | `aging_gamma` | 2⁻¹⁶ |
| Quorum minimum | q_min | `quorum_min` | 1 |
| Fan-out | — | `fan_out` | 3 |
| Max txs/vertex | — | `max_txs_per_vertex` | 64 |

### Notes

- **VDF difficulty**: The benchmarks use a low default (64) for tractable local execution. Increase for production-comparable timings (e.g., λ = 2¹⁸ targets ~200 ms on typical cloud vCPUs).
- **Deterministic seeding**: All experiments use `random.seed(42)` by default. Different seeds produce different absolute numbers but the same qualitative trends.
- **Simulation vs. distributed**: This implementation simulates multi-node behaviour in a single process. A distributed deployment across multiple machines will yield different absolute TPS numbers; relative comparisons between ChronoSeq and the baseline are preserved.

---

## How It Works

### VDF Epoch Beacon

```
Input:  previous epoch metadata (t-1, s_{t-1}, λ_{t-1}, Δ_{t-1})
        peer set P, quorum q_min, timing bounds (T_min, T_max)
Output: epoch seed s_t, difficulty λ_t, epoch length Δ_t

1. Evaluate VDF:  y ← VDFEval(s_{t-1}, λ_{t-1})
2. Broadcast VDF output and proof to peers
3. Collect endorsement signatures until quorum
4. If quorum reached:  s_t ← H(BEACON || t || y || qc)
   Else (degraded):    s_t ← H(BEACON_FALLBACK || t || y)
5. Retarget λ:
   - If T_vdf < 0.5·Δ:  increase λ
   - If T_vdf > 0.8·Δ:  decrease λ
6. Adjust Δ based on vertex backlog
```

### Ticket Assignment

```
Input:  DAG G=(V,E), ready set V_t, epoch seed s_t, rate caps ρ(·)
Output: ticket map, weight map

For each v ∈ V_t:
  1. Compute metadata:  meta[v] ← (height, inDeg, lane)
  2. Anti-grinding salt: salt[v] ← H(SALT || v.hash || v.txRoot || meta[v])
  3. Weight:  w[v] ← α if rate(origin(v)) > ρ(origin(v)), else 1
  4. Ticket:  ticket[v] ← Lexico(H(TICKET || s_t || salt[v]), v.hash)

Lane staggering:
  For each lane i:
    offset ← H(LANE || s_t || i)
    For each v in lane i:  ticket[v] ← Mix(ticket[v], offset)
```

### DAG Merge

```
Input:  DAG, ready set, tickets, weights, batch caps (B_max, T_max)
Output: ordered list L, batches {B_k}

1. Compute induced subgraph and in-degrees
2. Initialise min-heap with zero-indegree vertices, keyed by ticket
3. While heap not empty:
   a. Pop minimum-ticket vertex v → append to L and current batch
   b. Decrement in-degrees of v's children; push newly-free children
   c. Light aging: ticket[x] -= γ · k(x) for all x still in heap
   d. If |batch| ≥ B_max or elapsed ≥ T_max: emit batch, reset
4. Emit final partial batch
5. Defensive check: |L| must equal |V_t|
```

---

## Contributing

Contributions are welcome. Please open an issue or pull request.

## License

MIT — see [LICENSE](LICENSE).
