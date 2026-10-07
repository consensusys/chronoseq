#!/usr/bin/env python3
"""
ChronoSeq Test Suite — unit tests for all core components.

Run:  python -m pytest tests/ -v
  or: python tests/test_chronoseq.py
"""

import hashlib
import random
import struct
import time
import unittest
from typing import Dict, List, Set

from chronoseq.dag import DAGMempool, Transaction, Vertex, _merkle_root
from chronoseq.vdf import VDFBeacon, vdf_eval, vdf_verify
from chronoseq.merge import assign_tickets, dag_merge, TicketInfo
from chronoseq.adapters import EthereumAdapter, BitcoinAdapter, SolanaAdapter
from chronoseq.network import GossipNetwork
from chronoseq.network.baseline import SingleLeaderSequencer, BaselineTx


class TestMerkleRoot(unittest.TestCase):
    """Test Merkle root computation."""

    def test_empty(self):
        root = _merkle_root([])
        self.assertEqual(len(root), 32)

    def test_single_item(self):
        root = _merkle_root([b"hello"])
        self.assertEqual(len(root), 32)

    def test_deterministic(self):
        items = [b"a", b"b", b"c"]
        r1 = _merkle_root(items)
        r2 = _merkle_root(items)
        self.assertEqual(r1, r2)

    def test_order_matters(self):
        r1 = _merkle_root([b"a", b"b"])
        r2 = _merkle_root([b"b", b"a"])
        self.assertNotEqual(r1, r2)


class TestTransaction(unittest.TestCase):
    """Test Transaction hashing."""

    def test_hash_deterministic(self):
        tx = Transaction(sender="alice", payload=b"data", nonce=1)
        h1 = tx.tx_hash
        h2 = tx.tx_hash
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 32)

    def test_different_nonce_different_hash(self):
        tx1 = Transaction(sender="alice", payload=b"data", nonce=1)
        tx2 = Transaction(sender="alice", payload=b"data", nonce=2)
        self.assertNotEqual(tx1.tx_hash, tx2.tx_hash)


class TestVertex(unittest.TestCase):
    """Test Vertex hashing and properties."""

    def test_vertex_hash(self):
        txs = [Transaction(sender="alice", payload=b"p1", nonce=0)]
        v = Vertex(creator="node_0", lane=0, seq=0, txs=txs)
        self.assertEqual(len(v.vertex_hash), 32)

    def test_different_seq_different_hash(self):
        txs = [Transaction(sender="alice", payload=b"p1", nonce=0)]
        v1 = Vertex(creator="node_0", lane=0, seq=0, txs=txs)
        v2 = Vertex(creator="node_0", lane=0, seq=1, txs=txs)
        self.assertNotEqual(v1.vertex_hash, v2.vertex_hash)

    def test_height(self):
        v = Vertex(creator="node_0", lane=0, seq=5, txs=[])
        self.assertEqual(v.height, 5)


class TestDAGMempool(unittest.TestCase):
    """Test DAG mempool operations."""

    def test_create_vertex(self):
        mp = DAGMempool(num_lanes=4)
        txs = [Transaction(sender="alice", payload=b"p", nonce=0)]
        v = mp.create_vertex("node_0", lane=0, txs=txs)
        self.assertEqual(mp.total_vertices, 1)
        self.assertEqual(mp.total_transactions, 1)
        self.assertEqual(v.lane, 0)
        self.assertEqual(v.seq, 0)

    def test_multi_lane(self):
        mp = DAGMempool(num_lanes=4)
        for lane in range(4):
            txs = [Transaction(sender=f"node_{lane}", payload=b"p", nonce=0)]
            mp.create_vertex(f"node_{lane}", lane=lane, txs=txs)
        self.assertEqual(mp.total_vertices, 4)

    def test_intra_lane_causality(self):
        mp = DAGMempool(num_lanes=2)
        txs = [Transaction(sender="alice", payload=b"p", nonce=0)]
        v1 = mp.create_vertex("node_0", lane=0, txs=txs)
        v2 = mp.create_vertex("node_0", lane=0, txs=txs)
        # v2 should reference v1 as parent
        self.assertIn(v1.vertex_hash, v2.parents)

    def test_cross_lane_causality(self):
        mp = DAGMempool(num_lanes=2)
        txs = [Transaction(sender="alice", payload=b"p", nonce=0)]
        v1 = mp.create_vertex("node_0", lane=0, txs=txs)
        v2 = mp.create_vertex("node_1", lane=1, txs=txs, extra_parents=[v1.vertex_hash])
        self.assertIn(v1.vertex_hash, v2.parents)

    def test_induced_subgraph(self):
        mp = DAGMempool(num_lanes=2)
        txs = [Transaction(sender="alice", payload=b"p", nonce=0)]
        v1 = mp.create_vertex("node_0", lane=0, txs=txs)
        v2 = mp.create_vertex("node_0", lane=0, txs=txs)
        vs, edges = mp.get_induced_subgraph([v1, v2])
        self.assertEqual(len(vs), 2)
        self.assertIn(v1.vertex_hash, edges[v2.vertex_hash])

    def test_ready_set(self):
        mp = DAGMempool(num_lanes=2)
        txs = [Transaction(sender="alice", payload=b"p", nonce=0)]
        mp.create_vertex("node_0", lane=0, txs=txs)
        mp.create_vertex("node_1", lane=1, txs=txs)
        ready = mp.get_ready_set()
        self.assertEqual(len(ready), 2)


class TestVDFBeacon(unittest.TestCase):
    """Test VDF epoch beacon."""

    def test_vdf_eval_deterministic(self):
        seed = b"test_seed_1234567890abcdef1234567"
        y1 = vdf_eval(seed, 10)
        y2 = vdf_eval(seed, 10)
        self.assertEqual(y1, y2)

    def test_vdf_verify(self):
        seed = b"test_seed_1234567890abcdef1234567"
        y = vdf_eval(seed, 10)
        self.assertTrue(vdf_verify(seed, y, 10))

    def test_vdf_different_difficulty(self):
        seed = b"test_seed_1234567890abcdef1234567"
        y1 = vdf_eval(seed, 10)
        y2 = vdf_eval(seed, 20)
        self.assertNotEqual(y1, y2)

    def test_beacon_advance(self):
        beacon = VDFBeacon(initial_difficulty=4)
        m1 = beacon.advance_epoch()
        m2 = beacon.advance_epoch()
        self.assertEqual(m1.epoch_id, 1)
        self.assertEqual(m2.epoch_id, 2)
        self.assertNotEqual(m1.seed, m2.seed)
        self.assertGreater(m1.vdf_time_ms, 0)

    def test_beacon_degraded_mode(self):
        beacon = VDFBeacon(initial_difficulty=4, quorum_min=10)
        meta = beacon.advance_epoch(endorsements=1)
        self.assertTrue(meta.degraded)

    def test_difficulty_retargeting(self):
        beacon = VDFBeacon(initial_difficulty=4, epoch_length_ms=1.0)
        # VDF with difficulty=4 is extremely fast relative to epoch_length_ms=1ms
        # So ratio = vdf_time/epoch_length will be < 0.5, triggering an increase
        initial_diff = beacon.difficulty
        beacon.advance_epoch()
        # Difficulty should have been retargeted (either up or down)
        self.assertNotEqual(beacon.difficulty, initial_diff)


class TestTicketAssignment(unittest.TestCase):
    """Test ticket assignment."""

    def _make_vertices(self, n: int = 10) -> List[Vertex]:
        mp = DAGMempool(num_lanes=2)
        vertices = []
        for i in range(n):
            txs = [Transaction(sender=f"user_{i}", payload=bytes([i] * 32), nonce=i)]
            v = mp.create_vertex(f"node_{i % 2}", lane=i % 2, txs=txs)
            vertices.append(v)
        return vertices

    def test_assigns_all_vertices(self):
        verts = self._make_vertices(10)
        seed = b"epoch_seed_" + b"\x00" * 21
        tickets = assign_tickets(verts, seed)
        self.assertEqual(len(tickets), 10)

    def test_tickets_in_range(self):
        verts = self._make_vertices(100)
        seed = b"epoch_seed_" + b"\x00" * 21
        tickets = assign_tickets(verts, seed)
        for ti in tickets.values():
            self.assertGreaterEqual(ti.ticket, 0.0)
            self.assertLess(ti.ticket, 1.0)

    def test_deterministic(self):
        verts = self._make_vertices(10)
        seed = b"epoch_seed_" + b"\x00" * 21
        t1 = assign_tickets(verts, seed)
        t2 = assign_tickets(verts, seed)
        for vh in t1:
            self.assertEqual(t1[vh].ticket, t2[vh].ticket)

    def test_different_seed_different_tickets(self):
        verts = self._make_vertices(10)
        t1 = assign_tickets(verts, b"seed_A_" + b"\x00" * 25)
        t2 = assign_tickets(verts, b"seed_B_" + b"\x00" * 25)
        # At least some tickets should differ
        diffs = sum(1 for vh in t1 if t1[vh].ticket != t2[vh].ticket)
        self.assertGreater(diffs, 0)

    def test_rate_cap_weighting(self):
        verts = self._make_vertices(10)
        seed = b"epoch_seed_" + b"\x00" * 21
        tickets = assign_tickets(
            verts, seed,
            rate_caps={"node_000": 5.0},
            submit_rates={"node_000": 100.0},
            weight_penalty=0.3,
        )
        # Vertices from node_000 (lane 0) should have reduced weight
        for v in verts:
            if v.creator == "node_000":
                self.assertAlmostEqual(tickets[v.vertex_hash].weight, 0.3)


class TestDAGMerge(unittest.TestCase):
    """Test ticket-based DAG merge."""

    def test_basic_merge(self):
        mp = DAGMempool(num_lanes=2)
        txs = [Transaction(sender="a", payload=b"p", nonce=0)]
        v1 = mp.create_vertex("n0", lane=0, txs=txs)
        v2 = mp.create_vertex("n1", lane=1, txs=txs)

        ready = [v1, v2]
        _, edges = mp.get_induced_subgraph(ready)
        seed = b"epoch_seed_" + b"\x00" * 21
        tickets = assign_tickets(ready, seed)

        ordered, batches = dag_merge(ready, edges, tickets)
        self.assertEqual(len(ordered), 2)
        self.assertGreater(len(batches), 0)

    def test_respects_causality(self):
        """Parent must appear before child in the total order."""
        mp = DAGMempool(num_lanes=1)
        txs = [Transaction(sender="a", payload=b"p", nonce=0)]
        v1 = mp.create_vertex("n0", lane=0, txs=txs)
        v2 = mp.create_vertex("n0", lane=0, txs=txs)  # v2 → v1

        ready = [v1, v2]
        _, edges = mp.get_induced_subgraph(ready)
        seed = b"epoch_seed_" + b"\x00" * 21
        tickets = assign_tickets(ready, seed)

        ordered, _ = dag_merge(ready, edges, tickets)
        idx1 = ordered.index(v1.vertex_hash)
        idx2 = ordered.index(v2.vertex_hash)
        self.assertLess(idx1, idx2, "Parent must come before child")

    def test_batch_cutting(self):
        mp = DAGMempool(num_lanes=4)
        for lane in range(4):
            for _ in range(5):
                txs = [Transaction(sender="a", payload=b"p", nonce=random.randint(0, 10**6))]
                mp.create_vertex(f"n{lane}", lane=lane, txs=txs)

        ready = mp.get_ready_set()
        _, edges = mp.get_induced_subgraph(ready)
        seed = b"epoch_seed_" + b"\x00" * 21
        tickets = assign_tickets(ready, seed)

        ordered, batches = dag_merge(ready, edges, tickets, batch_max=5)
        total_in_batches = sum(b.size for b in batches)
        self.assertEqual(total_in_batches, len(ordered))
        # With 20 vertices and batch_max=5, should have ≥4 batches
        self.assertGreaterEqual(len(batches), 4)

    def test_all_vertices_ordered(self):
        mp = DAGMempool(num_lanes=8)
        for lane in range(8):
            txs = [Transaction(sender="a", payload=b"p", nonce=lane)]
            mp.create_vertex(f"n{lane}", lane=lane, txs=txs)

        ready = mp.get_ready_set()
        _, edges = mp.get_induced_subgraph(ready)
        seed = b"epoch_seed_" + b"\x00" * 21
        tickets = assign_tickets(ready, seed)

        ordered, _ = dag_merge(ready, edges, tickets)
        self.assertEqual(len(ordered), 8)


class TestAdapters(unittest.TestCase):
    """Test L1 settlement adapters."""

    def _make_batch(self):
        from chronoseq.merge import Batch
        return Batch(batch_id=0, vertices=[b"\x00" * 32], digest=b"\x01" * 32)

    def test_ethereum(self):
        adapter = EthereumAdapter()
        receipt = adapter.commit_batch(self._make_batch())
        self.assertEqual(receipt.adapter_name, "Ethereum")
        self.assertGreater(receipt.gas_used, 0)
        self.assertEqual(len(receipt.l1_tx_hash), 32)

    def test_bitcoin(self):
        adapter = BitcoinAdapter()
        receipt = adapter.commit_batch(self._make_batch())
        self.assertEqual(receipt.adapter_name, "Bitcoin")

    def test_solana(self):
        adapter = SolanaAdapter()
        receipt = adapter.commit_batch(self._make_batch())
        self.assertEqual(receipt.adapter_name, "Solana")


class TestGossipNetwork(unittest.TestCase):
    """Test simulated gossip network."""

    def test_network_creation(self):
        net = GossipNetwork(num_nodes=8)
        self.assertEqual(len(net.nodes), 8)
        self.assertEqual(net.mempool.num_lanes, 8)

    def test_simulate_round(self):
        net = GossipNetwork(num_nodes=4)
        vertices = net.simulate_round(txs_per_node=16)
        self.assertEqual(len(vertices), 4)
        self.assertEqual(net.mempool.total_vertices, 4)

    def test_gossip_latency_scales(self):
        net4 = GossipNetwork(num_nodes=4)
        net32 = GossipNetwork(num_nodes=32)
        # Average over samples — larger network should have higher latency
        lats4 = [net4.gossip_latency() for _ in range(100)]
        lats32 = [net32.gossip_latency() for _ in range(100)]
        # log(32) > log(4) so average should be higher
        import statistics
        self.assertGreater(statistics.mean(lats32), statistics.mean(lats4))


class TestSingleLeaderBaseline(unittest.TestCase):
    """Test single-leader baseline sequencer."""

    def test_fifo_ordering(self):
        sl = SingleLeaderSequencer(batch_size=10)
        txs = [BaselineTx(sender=f"u{i}", payload=b"p", nonce=i) for i in range(10)]
        sl.submit(txs)
        batch = sl.produce_batch()
        self.assertIsNotNone(batch)
        self.assertEqual(len(batch.tx_hashes), 10)

    def test_jains_index(self):
        sl = SingleLeaderSequencer(batch_size=100)
        txs = [BaselineTx(sender=f"u{i % 10}", payload=b"p", nonce=i) for i in range(100)]
        sl.submit(txs)
        sl.produce_batch()
        j = sl.jains_index()
        self.assertGreater(j, 0)
        self.assertLessEqual(j, 1.0)


class TestEndToEnd(unittest.TestCase):
    """End-to-end integration test: full ChronoSeq pipeline."""

    def test_full_pipeline(self):
        num_lanes = 4
        mp = DAGMempool(num_lanes=num_lanes)
        beacon = VDFBeacon(initial_difficulty=4)

        # Epoch 1: create vertices across lanes
        for lane in range(num_lanes):
            txs = [Transaction(sender=f"user_{lane}", payload=b"hello", nonce=lane)]
            mp.create_vertex(f"node_{lane}", lane=lane, txs=txs)

        # VDF beacon
        meta = beacon.advance_epoch(endorsements=3)
        self.assertFalse(meta.degraded)

        # Ticket assignment
        ready = mp.get_ready_set()
        tickets = assign_tickets(ready, meta.seed)
        self.assertEqual(len(tickets), num_lanes)

        # DAG merge
        _, edges = mp.get_induced_subgraph(ready)
        ordered, batches = dag_merge(ready, edges, tickets)
        self.assertEqual(len(ordered), num_lanes)

        # L1 commitment
        eth = EthereumAdapter()
        for batch in batches:
            receipt = eth.commit_batch(batch)
            self.assertEqual(receipt.adapter_name, "Ethereum")

        # Verify epoch 2 works
        for lane in range(num_lanes):
            txs = [Transaction(sender=f"user_{lane}", payload=b"world", nonce=lane + 10)]
            mp.create_vertex(f"node_{lane}", lane=lane, txs=txs)

        meta2 = beacon.advance_epoch(endorsements=3)
        self.assertEqual(meta2.epoch_id, 2)
        self.assertNotEqual(meta2.seed, meta.seed)


if __name__ == "__main__":
    unittest.main()
