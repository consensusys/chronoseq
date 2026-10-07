"""
Single-Leader Baseline Sequencer.

A FIFO sequencer that models the architectural pattern of centralized
rollups like Arbitrum and Optimism.  Used as the comparison baseline
in all benchmarks.

Properties:
  - Single designated leader node
  - Linear queue (FIFO ordering)
  - Same batch parameters as ChronoSeq for fair comparison
  - No fairness mechanism: transactions ordered by arrival time
"""

from __future__ import annotations

import hashlib
import random
import struct
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


def _sha256(*parts: bytes) -> bytes:
    h = hashlib.sha256()
    for p in parts:
        h.update(p)
    return h.digest()


@dataclass
class BaselineTx:
    """Transaction in the single-leader baseline."""

    sender: str
    payload: bytes
    nonce: int = 0
    arrived_at: float = field(default_factory=time.time)

    @property
    def tx_hash(self) -> bytes:
        return _sha256(
            self.sender.encode(),
            self.payload,
            struct.pack(">Q", self.nonce),
        )


@dataclass
class BaselineBatch:
    """Batch produced by the single-leader."""

    batch_id: int
    tx_hashes: List[bytes]
    digest: bytes
    timestamp: float = field(default_factory=time.time)


class SingleLeaderSequencer:
    """
    Centralized FIFO sequencer baseline.

    All transactions enter a single queue and are batched in FIFO order.
    No fairness randomisation is applied.

    Parameters match ChronoSeq's for fair comparison:
      - batch_size: B_max
      - epoch_length_ms: Δ
    """

    def __init__(self, batch_size: int = 256, epoch_length_ms: float = 500.0):
        self.batch_size = batch_size
        self.epoch_length_ms = epoch_length_ms
        self.queue: List[BaselineTx] = []
        self.batches: List[BaselineBatch] = []
        self._batch_id = 0
        self.per_sender_wait: Dict[str, List[float]] = {}

    def submit(self, txs: List[BaselineTx]):
        """Submit transactions to the leader's queue."""
        self.queue.extend(txs)

    def produce_batch(self) -> Optional[BaselineBatch]:
        """Produce a FIFO-ordered batch from the queue."""
        if not self.queue:
            return None

        now = time.time()
        batch_txs = self.queue[: self.batch_size]
        self.queue = self.queue[self.batch_size :]

        tx_hashes = [tx.tx_hash for tx in batch_txs]
        digest = _sha256(*tx_hashes)

        # Record per-sender waiting times
        for tx in batch_txs:
            wait = now - tx.arrived_at
            self.per_sender_wait.setdefault(tx.sender, []).append(wait)

        batch = BaselineBatch(
            batch_id=self._batch_id,
            tx_hashes=tx_hashes,
            digest=digest,
        )
        self.batches.append(batch)
        self._batch_id += 1
        return batch

    def jains_index(self) -> float:
        """
        Compute Jain's fairness index over per-sender average waiting times.

        J = (Σ W_u)² / (U · Σ W_u²)
        """
        if not self.per_sender_wait:
            return 1.0

        avg_waits = []
        for sender, waits in self.per_sender_wait.items():
            avg_waits.append(sum(waits) / len(waits))

        if not avg_waits:
            return 1.0

        U = len(avg_waits)
        sum_w = sum(avg_waits)
        sum_w2 = sum(w * w for w in avg_waits)

        if sum_w2 == 0:
            return 1.0

        return (sum_w ** 2) / (U * sum_w2)
