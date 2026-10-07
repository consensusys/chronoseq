"""ChronoSeq WAN timing simulator.

The simulator models the ordering layer of ChronoSeq and of the baselines used in
the paper at micro-block granularity:

* a three-region WAN (Sydney / Singapore / Oregon) with measured inter-region
  RTTs, multiplicative log-normal jitter and optional latency spikes;
* per-node uplink and downlink serialization for every payload-carrying message
  (micro-blocks, FIFO batches), using the Lindley recursion;
* lane dissemination with availability certificates (2f+1 acks);
* a two-phase, PBFT-style agreement on a compact per-lane cut with rotating
  proposers, a floor (inclusion) rule and view changes;
* VDF-seeded per-sender (or per-vertex) tickets, deterministic DAG order,
  median receive-timestamp order, a rotating-leader BFT sequencer and a
  single-leader FIFO sequencer;
* Byzantine behaviours: withholding proposals, silent voters, censorship,
  exclusion by faulty leaders, faulty ingress, informed vetoes, and frontrunning
  by injection (equivocation only in the naive local-ready-set analysis).

Not modeled: CPU (except a per-transaction ordering cost; bench/ measures the
rest), bandwidth of control messages, VDF proof generation (fixed delay,
unlimited concurrency), view-change internals (a failed view costs vc_timeout;
no new-view certificates or re-proposals), certificate forwarding, (sender,
nonce) de-duplication and nonce carry-over, client signatures, equivocating
leaders. The local lock deadline is modeled in view 0. The agreed cut is
computed once per epoch for all nodes, so consistency of the agreed design
follows from the protocol (Theorem 1 in the paper), not from simulation.

All times are in seconds.
"""
