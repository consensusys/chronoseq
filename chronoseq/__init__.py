"""
ChronoSeq: A Decentralized Layer-2 Sequencer with DAG Mempool and VDF Ordering.

Modules:
  dag       — DAG mempool with multi-lane vertices and causality edges
  vdf       — VDF epoch beacon (Pietrzak-style sequential squaring)
  merge     — Ticket assignment and ticket-based DAG merge
  adapters  — L1 settlement adapter stubs (Ethereum / Bitcoin / Solana)
  network   — Simulated gossip transport for multi-node benchmarks
"""

__version__ = "0.1.0"
