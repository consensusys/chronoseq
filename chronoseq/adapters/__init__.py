"""
L1 Settlement Adapters.

ChronoSeq abstracts settlement from ordering via per-L1 adapter modules.
Each adapter translates a ChronoSeq batch digest into a valid L1 transaction.

This module provides:
  - BaseAdapter: abstract interface
  - EthereumAdapter: calldata commitment (simulated)
  - BitcoinAdapter: OP_RETURN digest (simulated)
  - SolanaAdapter: instruction packet (simulated)
"""

from __future__ import annotations

import hashlib
import struct
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from chronoseq.merge import Batch


def _sha256(*parts: bytes) -> bytes:
    h = hashlib.sha256()
    for p in parts:
        h.update(p)
    return h.digest()


@dataclass
class CommitmentReceipt:
    """Receipt from an L1 commitment."""

    adapter_name: str
    batch_id: int
    digest: bytes
    l1_tx_hash: bytes
    timestamp: float
    gas_used: int = 0
    block_number: int = 0


class BaseAdapter(ABC):
    """Abstract L1 settlement adapter."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def commit_batch(self, batch: Batch) -> CommitmentReceipt:
        """Commit a batch digest to the L1."""
        ...


class EthereumAdapter(BaseAdapter):
    """
    Ethereum calldata adapter.

    Simulates submitting batch digest as rollup calldata on Ethereum.
    In production: encodes digest into a transaction sent to the rollup
    inbox contract, with gas costs proportional to data size.
    """

    def __init__(self, base_gas: int = 21_000, per_byte_gas: int = 16):
        self.base_gas = base_gas
        self.per_byte_gas = per_byte_gas
        self._block = 1_000_000
        self.committed: List[CommitmentReceipt] = []

    @property
    def name(self) -> str:
        return "Ethereum"

    def commit_batch(self, batch: Batch) -> CommitmentReceipt:
        # Calldata: 32-byte digest + 4-byte batch_id
        calldata_size = 36
        gas = self.base_gas + calldata_size * self.per_byte_gas
        self._block += 1
        tx_hash = _sha256(b"ETH_TX", batch.digest, struct.pack(">I", batch.batch_id))

        receipt = CommitmentReceipt(
            adapter_name=self.name,
            batch_id=batch.batch_id,
            digest=batch.digest,
            l1_tx_hash=tx_hash,
            timestamp=time.time(),
            gas_used=gas,
            block_number=self._block,
        )
        self.committed.append(receipt)
        return receipt


class BitcoinAdapter(BaseAdapter):
    """
    Bitcoin OP_RETURN adapter.

    Simulates embedding a 32-byte batch digest in an OP_RETURN output.
    Bitcoin's OP_RETURN limit is 80 bytes, sufficient for our digest.
    """

    def __init__(self):
        self._block = 850_000
        self.committed: List[CommitmentReceipt] = []

    @property
    def name(self) -> str:
        return "Bitcoin"

    def commit_batch(self, batch: Batch) -> CommitmentReceipt:
        self._block += 1
        tx_hash = _sha256(
            b"BTC_TX", batch.digest, struct.pack(">I", batch.batch_id)
        )
        receipt = CommitmentReceipt(
            adapter_name=self.name,
            batch_id=batch.batch_id,
            digest=batch.digest,
            l1_tx_hash=tx_hash,
            timestamp=time.time(),
            gas_used=0,
            block_number=self._block,
        )
        self.committed.append(receipt)
        return receipt


class SolanaAdapter(BaseAdapter):
    """
    Solana instruction packet adapter.

    Simulates sending a batch digest as a Solana instruction to a
    program account.
    """

    def __init__(self):
        self._slot = 200_000_000
        self.committed: List[CommitmentReceipt] = []

    @property
    def name(self) -> str:
        return "Solana"

    def commit_batch(self, batch: Batch) -> CommitmentReceipt:
        self._slot += 1
        tx_hash = _sha256(
            b"SOL_TX", batch.digest, struct.pack(">I", batch.batch_id)
        )
        receipt = CommitmentReceipt(
            adapter_name=self.name,
            batch_id=batch.batch_id,
            digest=batch.digest,
            l1_tx_hash=tx_hash,
            timestamp=time.time(),
            gas_used=0,
            block_number=self._slot,
        )
        self.committed.append(receipt)
        return receipt
