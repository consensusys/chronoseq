"""Class-group VDF cost on this machine, using chiavdf (Wesolowski proofs).

Measures:
  1. iterations per second of evaluation plus proof generation (calibration);
  2. wall time and verification time at the smallest difficulties lambda_min
     for which even the fastest observed rate needs more than A_max * W
     (W = 0.84 s, A_max = 1 and 2), with a 2% margin;
  3. per-evaluation time with 1 or 2 evaluations running concurrently
     (the epoch pipeline runs ceil(T_VDF / cut interval) of them).

Usage: python bench/vdf_bench.py > results/bench_vdf.csv
"""
import math
import os
import time
from multiprocessing import Pool

import chiavdf

BITS = 1024
X0 = b"\x08" + b"\x00" * 99  # class-group generator form (100 bytes)


def prove(it):
    ch = os.urandom(32)
    t = time.perf_counter()
    r = chiavdf.prove(ch, X0, BITS, it, "")
    dt = time.perf_counter() - t
    fs = len(r) // 2
    d = chiavdf.create_discriminant(ch, BITS)
    t = time.perf_counter()
    ok = chiavdf.verify_wesolowski(str(d), X0, r[:fs], r[fs:], it)
    vt = time.perf_counter() - t
    assert ok
    return dt, vt


def main():
    print("kind,target_s,iters,conc,rep,prove_s,verify_ms", flush=True)
    cal = []
    for rep in range(10):
        dt, vt = prove(200_000)
        cal.append(200_000 / dt)
        print(f"calib,,200000,1,{rep},{dt:.4f},{vt * 1000:.3f}", flush=True)
    # lambda_min: even at the fastest observed rate, evaluation must take longer
    # than A_max * W (W = 0.84 s), with a 2% margin
    r_max = max(cal)
    for target in (0.84, 1.68):
        it = int(math.ceil(r_max * target * 1.02 / 1000.0)) * 1000
        for rep in range(10):
            dt, vt = prove(it)
            print(f"target,{target},{it},1,{rep},{dt:.4f},{vt * 1000:.3f}", flush=True)
    it = int(math.ceil(r_max * 0.84 * 1.02 / 1000.0)) * 1000
    for conc in (1, 2):
        with Pool(conc) as p:
            for rep in range(5):
                for dt, vt in p.map(prove, [it] * conc):
                    print(f"conc,0.84,{it},{conc},{rep},{dt:.4f},{vt * 1000:.3f}", flush=True)


if __name__ == "__main__":
    main()
