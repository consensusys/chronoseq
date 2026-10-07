#!/usr/bin/env bash
# CPU microbenchmarks used for the per-node CPU budget in the paper.
set -euo pipefail
cd "$(dirname "$0")/.."
gcc -O2 -Wall -o bench/order_bench bench/order_bench.c -lcrypto
{ echo "m,rep,ns_per_tx,ns_order_per_tx,senders,kept,check"
  for m in 10000 100000 1000000; do bench/order_bench "$m" 5; done; } > results/bench_order.csv
python bench/bls_bench.py > results/bench_bls.csv
python bench/sig_bench.py > results/bench_sig.csv
python bench/vdf_bench.py > results/bench_vdf.csv
python -m bench.sim_rates > results/bench_rates.csv
echo "benchmarks -> results/bench_*.csv"
