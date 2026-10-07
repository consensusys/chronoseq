"""
DAG Mempool — Multi-lane directed acyclic graph of transaction vertices.

Features:
  - Transaction vertices (micro-blocks) with Merkle commitments
  - Causality edges encoding happened-before relationships
  - Multi-lane structure for parallel dissemination
  - Gossip-based propagation across independent lanes
"""

from __future__ import annotations

import hashlib
import struct
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


def _sha256(*parts: bytes) -> bytes:
    """Compute SHA-256 over concatenated byte parts."""
    h = hashlib.sha256()
    for p in parts:
        h.update(p)
    return h.digest()


def _merkle_root(items: List[bytes]) -> bytes:
    """Compute a simple binary Merkle root over a list of byte items."""
    if not items:
        return b"\x00" * 32
    layer = [hashlib.sha256(item).digest() for item in items]
    while len(layer) > 1:
        next_layer = []
        for i in range(0, len(layer), 2):
            if i + 1 < len(layer):
                next_layer.append(_sha256(layer[i], layer[i + 1]))
            else:
                next_layer.append(layer[i])
        layer = next_layer
    return layer[0]


@dataclass
class Transaction:
    """A single client transaction."""

    sender: str
    payload: bytes
    nonce: int = 0
    timestamp: float = field(default_factory=time.time)

    @property
    def tx_hash(self) -> bytes:
        return _sha256(
            self.sender.encode(),
            self.payload,
            struct.pack(">Q", self.nonce),
        )


@dataclass
class Vertex:
    """
    A DAG vertex (micro-block).

    Each vertex encapsulates a cryptographically signed bundle of transactions
    together with a Merkle commitment.  It records its creator node identity
    and a monotone sequence number so that two vertices from the same node are
    totally ordered.

    Attributes:
        creator     — node ID that produced this vertex
        lane        — lane index (each node has one dedicated lane)
        seq         — monotone sequence number within lane
        txs         — bundled transactions
        parents     — hashes of parent vertices (causality edges)
        created_at  — wall-clock creation time (seconds)
    """

    creator: str
    lane: int
    seq: int
    txs: List[Transaction]
    parents: List[bytes] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    # Computed once and cached
    _hash_cache: Optional[bytes] = field(default=None, repr=False)
    _tx_root_cache: Optional[bytes] = field(default=None, repr=False)

    @property
    def tx_root(self) -> bytes:
        """Merkle root of all transaction hashes in this vertex."""
        if self._tx_root_cache is None:
            self._tx_root_cache = _merkle_root([tx.tx_hash for tx in self.txs])
        return self._tx_root_cache

    @property
    def vertex_hash(self) -> bytes:
        """
        Unique hash of the vertex.

        Covers: creator, lane, seq, tx_root, and parent hashes.
        """
        if self._hash_cache is None:
            parts = [
                self.creator.encode(),
                struct.pack(">II", self.lane, self.seq),
                self.tx_root,
            ]
            for p in sorted(self.parents):
                parts.append(p)
            self._hash_cache = _sha256(*parts)
        return self._hash_cache

    @property
    def height(self) -> int:
        """Logical height — same as seq for single-lane vertices."""
        return self.seq

    def __hash__(self):
        return int.from_bytes(self.vertex_hash[:8], "big")

    def __eq__(self, other):
        if not isinstance(other, Vertex):
            return NotImplemented
        return self.vertex_hash == other.vertex_hash


class DAGMempool:
    """
    Multi-lane DAG mempool.

    Each node maintains one lane — a sequence of micro-blocks it produces.
    Lanes are independent, so aggregate ingestion bandwidth scales with
    the number of active nodes.

    The DAG mempool provides:
      - Availability: multiple gossip paths per vertex
      - Parallelism: lanes disseminated simultaneously
    """

    def __init__(self, num_lanes: int, max_txs_per_vertex: int = 64):
        self.num_lanes = num_lanes
        self.max_txs_per_vertex = max_txs_per_vertex

        # vertex_hash → Vertex
        self.vertices: Dict[bytes, Vertex] = {}
        # child_hash → set of parent_hashes  (causality edges u → v)
        self.edges: Dict[bytes, Set[bytes]] = {}
        # lane_index → list of vertex hashes in sequence order
        self.lanes: Dict[int, List[bytes]] = {i: [] for i in range(num_lanes)}
        # lane_index → next sequence number
        self._seq_counters: Dict[int, int] = {i: 0 for i in range(num_lanes)}

    # ---- Vertex creation -------------------------------------------------

    def create_vertex(
        self,
        creator: str,
        lane: int,
        txs: List[Transaction],
        extra_parents: Optional[List[bytes]] = None,
    ) -> Vertex:
        """
        Create a new vertex in a given lane.

        Automatically references the previous vertex in the same lane
        (if any) and any additional cross-lane parents.
        """
        if lane < 0 or lane >= self.num_lanes:
            raise ValueError(f"lane {lane} out of range [0, {self.num_lanes})")

        seq = self._seq_counters[lane]
        parents: List[bytes] = []

        # Reference the tip of the same lane (intra-lane causality)
        if self.lanes[lane]:
            parents.append(self.lanes[lane][-1])

        # Cross-lane causality edges
        if extra_parents:
            for p in extra_parents:
                if p in self.vertices and p not in parents:
                    parents.append(p)

        vertex = Vertex(
            creator=creator,
            lane=lane,
            seq=seq,
            txs=txs[:self.max_txs_per_vertex],
            parents=parents,
        )

        self._insert(vertex)
        return vertex

    def _insert(self, vertex: Vertex):
        """Insert a vertex into the mempool."""
        vh = vertex.vertex_hash
        self.vertices[vh] = vertex
        self.edges[vh] = set(vertex.parents)
        self.lanes[vertex.lane].append(vh)
        self._seq_counters[vertex.lane] = vertex.seq + 1

    # ---- Ready set extraction --------------------------------------------

    def get_ready_set(self) -> List[Vertex]:
        """
        Return all vertices currently in the DAG — the 'ready set' for the
        current epoch.

        In a full implementation the ready set would be the vertices added
        since the last epoch boundary.  Here we return everything for
        simplicity; the benchmark harness manages epoch boundaries.
        """
        return list(self.vertices.values())

    def get_epoch_ready_set(self, since_hashes: Set[bytes]) -> List[Vertex]:
        """Return vertices added after a given set of already-committed hashes."""
        return [
            v
            for h, v in self.vertices.items()
            if h not in since_hashes
        ]

    # ---- Graph queries ---------------------------------------------------

    def get_children(self, vertex_hash: bytes) -> List[Vertex]:
        """Return vertices that directly reference vertex_hash as a parent."""
        children = []
        for vh, parents in self.edges.items():
            if vertex_hash in parents:
                children.append(self.vertices[vh])
        return children

    def indegree(self, vertex_hash: bytes) -> int:
        """Number of parents for a given vertex."""
        return len(self.edges.get(vertex_hash, set()))

    def get_induced_subgraph(
        self, vertex_set: List[Vertex]
    ) -> Tuple[List[Vertex], Dict[bytes, Set[bytes]]]:
        """
        Return the induced subgraph over a set of vertices —
        only edges where both endpoints are in the set.
        """
        vset = {v.vertex_hash for v in vertex_set}
        edges: Dict[bytes, Set[bytes]] = {}
        for v in vertex_set:
            edges[v.vertex_hash] = {p for p in v.parents if p in vset}
        return vertex_set, edges

    # ---- Stats -----------------------------------------------------------

    @property
    def total_vertices(self) -> int:
        return len(self.vertices)

    @property
    def total_transactions(self) -> int:
        return sum(len(v.txs) for v in self.vertices.values())

    def lane_depths(self) -> Dict[int, int]:
        return {lane: len(hashes) for lane, hashes in self.lanes.items()}
