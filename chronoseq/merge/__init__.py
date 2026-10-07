"""
Ticket Assignment and DAG Merge.

Ticket Assignment:
  - Anti-grinding salt: salt[v] = H(SALT || v.hash || v.txRoot || meta[v])
  - Per-origin rate-cap weighting
  - Ticket: H(TICKET || s_t || salt[v])  with hash tie-break
  - Lane staggering via Mix(ticket, offset)

Ticket-based DAG Merge:
  - Kahn's topological sort with min-heap keyed by ticket
  - Light aging: ticket[x] -= γ * k(x) per merge step
  - Batch cutting by size or elapsed time
"""

from __future__ import annotations

import hashlib
import heapq
import struct
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from chronoseq.dag import Vertex


def _sha256(*parts: bytes) -> bytes:
    h = hashlib.sha256()
    for p in parts:
        h.update(p)
    return h.digest()


def _ticket_to_float(ticket_bytes: bytes) -> float:
    """Convert a 32-byte ticket hash to a float in [0, 1)."""
    # Use first 8 bytes for enough precision
    val = int.from_bytes(ticket_bytes[:8], "big")
    return val / (2**64)


# ---------------------------------------------------------------------------
#  Ticket Assignment
# ---------------------------------------------------------------------------

# Domain separation tags
TAG_SALT = b"SALT"
TAG_TICKET = b"TICKET"
TAG_LANE = b"LANE"


@dataclass
class TicketInfo:
    """Ticket and weight for a single vertex."""

    vertex_hash: bytes
    ticket: float        # normalised [0, 1)
    ticket_bytes: bytes  # raw 32-byte hash for tie-breaking
    weight: float = 1.0


def assign_tickets(
    ready_set: List[Vertex],
    epoch_seed: bytes,
    rate_caps: Optional[Dict[str, float]] = None,
    submit_rates: Optional[Dict[str, float]] = None,
    weight_penalty: float = 0.5,
) -> Dict[bytes, TicketInfo]:
    """
    Assign VDF-derived tickets to DAG vertices.

    Parameters:
        ready_set       — list of vertices in the epoch's ready set V_t
        epoch_seed      — s_t (32 bytes) from VDF beacon
        rate_caps       — per-origin rate cap ρ(u);  None = no rate limiting
        submit_rates    — per-origin recent submit rate r_u
        weight_penalty  — α ∈ (0,1) applied when r_u > ρ(u)

    Returns:
        vertex_hash → TicketInfo  mapping
    """
    tickets: Dict[bytes, TicketInfo] = {}

    # --- Lines 3-4: compute metadata and anti-grinding salt ---
    for v in ready_set:
        meta = struct.pack(">III", v.height, len(v.parents), v.lane)
        salt = _sha256(TAG_SALT, v.vertex_hash, v.tx_root, meta)

        # --- Lines 5-8: per-origin rate-cap weighting ---
        w = 1.0
        if rate_caps and submit_rates:
            origin = v.creator
            cap = rate_caps.get(origin, float("inf"))
            rate = submit_rates.get(origin, 0.0)
            if rate > cap:
                w = weight_penalty

        # --- Lines 9-11: ticket computation with hash tie-break ---
        h0 = _sha256(TAG_TICKET, epoch_seed, salt)
        # Lexicographic pair (h0, v.hash) — we just hash them together
        ticket_bytes = _sha256(h0, v.vertex_hash)
        ticket_float = _ticket_to_float(ticket_bytes)

        tickets[v.vertex_hash] = TicketInfo(
            vertex_hash=v.vertex_hash,
            ticket=ticket_float,
            ticket_bytes=ticket_bytes,
            weight=w,
        )

    # --- Lines 13-16: lane staggering via Mix ---
    lanes: Dict[int, List[Vertex]] = {}
    for v in ready_set:
        lanes.setdefault(v.lane, []).append(v)

    for lane_id, lane_vertices in lanes.items():
        offset = _sha256(TAG_LANE, epoch_seed, struct.pack(">I", lane_id))
        for v in lane_vertices:
            ti = tickets[v.vertex_hash]
            mixed = _sha256(ti.ticket_bytes, offset)
            ti.ticket_bytes = mixed
            ti.ticket = _ticket_to_float(mixed)

    return tickets


# ---------------------------------------------------------------------------
#  Ticket-based DAG Merge
# ---------------------------------------------------------------------------

@dataclass
class Batch:
    """A committed batch of ordered vertices."""

    batch_id: int
    vertices: List[bytes]   # ordered vertex hashes
    digest: bytes           # Merkle root of the batch

    @property
    def size(self) -> int:
        return len(self.vertices)


def dag_merge(
    ready_set: List[Vertex],
    edges: Dict[bytes, Set[bytes]],
    tickets: Dict[bytes, TicketInfo],
    batch_max: int = 256,
    time_max_ms: float = 500.0,
    aging_gamma: float = 2**-16,
) -> Tuple[List[bytes], List[Batch]]:
    """
    Ticket-based DAG Merge.

    Produces a globally consistent total order that respects causality
    and ranks vertices by VDF-derived tickets.  Light aging prevents
    starvation of older vertices.

    Parameters:
        ready_set     — vertices to order
        edges         — vertex_hash → set of parent hashes (induced subgraph)
        tickets       — vertex_hash → TicketInfo from ticket assignment
        batch_max     — B_max: maximum vertices per batch
        time_max_ms   — T_max: maximum batch duration in ms
        aging_gamma   — γ: aging factor (default 2^-16)

    Returns:
        (ordered_hashes, batches)
    """
    # --- Line 3: compute in-degrees over induced subgraph ---
    in_degree: Dict[bytes, int] = {}
    children: Dict[bytes, List[bytes]] = {}
    vh_set = {v.vertex_hash for v in ready_set}

    for v in ready_set:
        vh = v.vertex_hash
        in_degree[vh] = 0
        children[vh] = []

    for v in ready_set:
        vh = v.vertex_hash
        for parent in edges.get(vh, set()):
            if parent in vh_set:
                in_degree[vh] += 1
                children.setdefault(parent, []).append(vh)

    # --- Line 4-5: initialise min-heap with roots (in-degree 0) ---
    # Heap entries: (ticket, -weight, vertex_hash)
    heap: List[Tuple[float, float, bytes]] = []
    for v in ready_set:
        vh = v.vertex_hash
        if in_degree[vh] == 0:
            ti = tickets[vh]
            heapq.heappush(heap, (ti.ticket, -ti.weight, vh))

    # Track effective tickets for aging
    effective_ticket: Dict[bytes, float] = {
        vh: tickets[vh].ticket for vh in vh_set
    }
    wait_steps: Dict[bytes, int] = {vh: 0 for vh in vh_set}

    # --- Lines 7-20: main merge loop ---
    ordered: List[bytes] = []
    batches: List[Batch] = []
    current_batch: List[bytes] = []
    batch_id = 0
    t0 = time.perf_counter()
    merge_step = 0

    while heap:
        _, _, vh = heapq.heappop(heap)
        ordered.append(vh)
        current_batch.append(vh)

        # --- Lines 11-13: release children ---
        for child in children.get(vh, []):
            in_degree[child] -= 1
            if in_degree[child] == 0:
                ti = tickets[child]
                heapq.heappush(
                    heap,
                    (effective_ticket[child], -ti.weight, child),
                )

        # --- Lines 14-15: light aging ---
        merge_step += 1
        new_heap = []
        for entry in heap:
            _, neg_w, h = entry
            wait_steps[h] += 1
            aged = effective_ticket[h] - aging_gamma * wait_steps[h]
            effective_ticket[h] = aged
            new_heap.append((aged, neg_w, h))
        heapq.heapify(new_heap)
        heap = new_heap

        # --- Lines 16-19: batch cutting ---
        elapsed = (time.perf_counter() - t0) * 1000.0
        if len(current_batch) >= batch_max or elapsed >= time_max_ms:
            digest = _sha256(*(h for h in current_batch))
            batches.append(
                Batch(batch_id=batch_id, vertices=list(current_batch), digest=digest)
            )
            batch_id += 1
            current_batch = []
            t0 = time.perf_counter()

    # --- Line 21: emit final partial batch ---
    if current_batch:
        digest = _sha256(*(h for h in current_batch))
        batches.append(
            Batch(batch_id=batch_id, vertices=list(current_batch), digest=digest)
        )

    # --- Line 22: defensive check ---
    if len(ordered) < len(ready_set):
        raise RuntimeError(
            f"Merge incomplete: ordered {len(ordered)} of {len(ready_set)} vertices "
            f"— possible cycle in DAG"
        )

    return ordered, batches
