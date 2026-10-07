"""Vertex rate per lane and cut interval from the simulator, for the CPU budget.

The per-node CPU budget in the paper multiplies measured primitive costs
(bench_bls.csv, bench_order.csv, bench_vdf.csv) by these simulated rates.
Usage: python -m bench.sim_rates > results/bench_rates.csv
"""
import numpy as np

from sim.core import Config
from sim.runner import simulate_dag


def main():
    print("n,load,seed,vertex_rate_per_lane,cut_interval,p50_fast")
    for n in (32, 128):
        for load in (20000, 225000):
            for seed in range(1, 11):
                cfg = Config(n=n, load_tps=load, duration=8)
                run = simulate_dag(cfg, seed)
                cr = np.asarray(run.L.created)
                sel = (cr > cfg.warmup) & (cr < cfg.duration)
                vr = sel.sum() / (cfg.duration - cfg.warmup) / n
                iv = float(np.median(np.diff(run.Ep.P)))
                lat = run.latency(0.85)[run.w.submit > cfg.warmup]
                print(f"{n},{load},{seed},{vr:.3f},{iv:.4f},{np.median(lat):.4f}", flush=True)


if __name__ == "__main__":
    main()
