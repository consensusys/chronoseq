"""
Simulated Gossip Network.

Provides a lightweight simulation of gossip-based vertex dissemination
across a network of ChronoSeq nodes.  Each node maintains its own lane
and receives vertices from peers via simulated gossip with configurable
latency.

Default parameters:
  - Fan-out: 3
  - Gossip delay: 40-80 ms per round (modelled as O(log n))
  - Max transactions per micro-block: 64
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from chronoseq.dag import DAGMempool, Transaction, Vertex


@dataclass
class GossipMessage:
    """A message carrying a vertex through the gossip network."""

    vertex: Vertex
    sender: str
    hops: int = 0
    created_at: float = field(default_factory=time.time)


class SimulatedNode:
    """
    A single ChronoSeq sequencer node.

    Each node has a dedicated lane in the DAG mempool and participates
    in gossip dissemination.
    """

    def __init__(self, node_id: str, lane: int, mempool: DAGMempool):
        self.node_id = node_id
        self.lane = lane
        self.mempool = mempool
        self.received_hashes: Set[bytes] = set()
        self.tx_queue: List[Transaction] = []

    def submit_transactions(self, txs: List[Transaction]):
        """Queue client transactions for inclusion in the next micro-block."""
        self.tx_queue.extend(txs)

    def produce_vertex(
        self, cross_lane_parents: Optional[List[bytes]] = None
    ) -> Optional[Vertex]:
        """
        Produce a micro-block from queued transactions.

        References the tip of the node's lane plus any cross-lane parents
        to create causality edges.
        """
        if not self.tx_queue:
            return None

        txs = self.tx_queue[: self.mempool.max_txs_per_vertex]
        self.tx_queue = self.tx_queue[self.mempool.max_txs_per_vertex :]

        v = self.mempool.create_vertex(
            creator=self.node_id,
            lane=self.lane,
            txs=txs,
            extra_parents=cross_lane_parents,
        )
        self.received_hashes.add(v.vertex_hash)
        return v


class GossipNetwork:
    """
    Simulated gossip network for benchmarking.

    Models the propagation of DAG vertices across nodes with
    configurable fan-out and latency.
    """

    def __init__(
        self,
        num_nodes: int,
        fan_out: int = 3,
        base_latency_ms: float = 10.0,
        jitter_ms: float = 5.0,
    ):
        self.num_nodes = num_nodes
        self.fan_out = fan_out
        self.base_latency_ms = base_latency_ms
        self.jitter_ms = jitter_ms

        # Shared mempool (in simulation all nodes share state)
        self.mempool = DAGMempool(
            num_lanes=num_nodes, max_txs_per_vertex=64
        )

        # Create nodes
        self.nodes: Dict[str, SimulatedNode] = {}
        for i in range(num_nodes):
            node_id = f"node_{i:03d}"
            self.nodes[node_id] = SimulatedNode(node_id, i, self.mempool)

    def gossip_latency(self) -> float:
        """
        Simulated gossip delay in milliseconds.

        T_gossip = O(log n) — logarithmic in node count.
        """
        hops = math.ceil(math.log2(max(2, self.num_nodes)))
        per_hop = self.base_latency_ms + random.uniform(
            -self.jitter_ms, self.jitter_ms
        )
        return hops * max(0, per_hop)

    def simulate_round(
        self, txs_per_node: int = 32, cross_lane_prob: float = 0.2
    ) -> List[Vertex]:
        """
        Simulate one round of gossip dissemination.

        Each node:
          1. Queues synthetic transactions.
          2. Produces a micro-block.
          3. Gossip propagates (simulated by shared mempool).

        Returns the list of vertices produced in this round.
        """
        produced: List[Vertex] = []

        for node_id, node in self.nodes.items():
            # Generate synthetic transactions
            txs = [
                Transaction(
                    sender=f"client_{random.randint(0, 999):04d}",
                    payload=random.randbytes(64),
                    nonce=random.randint(0, 2**32),
                )
                for _ in range(txs_per_node)
            ]
            node.submit_transactions(txs)

            # Optionally reference a random cross-lane parent
            cross_parents = None
            if random.random() < cross_lane_prob and self.mempool.total_vertices > 0:
                other_lanes = [
                    l
                    for l in range(self.num_nodes)
                    if l != node.lane and self.mempool.lanes[l]
                ]
                if other_lanes:
                    pick_lane = random.choice(other_lanes)
                    cross_parents = [self.mempool.lanes[pick_lane][-1]]

            v = node.produce_vertex(cross_lane_parents=cross_parents)
            if v:
                produced.append(v)

        return produced
