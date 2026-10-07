"""BLS12-381 signature costs (blspy, proof-of-possession scheme) on this machine.

Measures signing, single verification, aggregation of q signatures and
fast aggregate verification against q public keys (multi-signature on one
message, as used for acknowledgments, availability certificates and votes),
for the quorum sizes of n = 32, 64 and 128.

Usage: python bench/bls_bench.py > results/bench_bls.csv
"""
import os
import time

from blspy import G2Element, PopSchemeMPL, PrivateKey

REPS = 200


def timeit(fn, reps):
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t) / reps * 1000  # ms


def main():
    print("op,q,ms", flush=True)
    sks = [PopSchemeMPL.key_gen(os.urandom(32)) for _ in range(86)]
    pks = [sk.get_g1() for sk in sks]
    msg = os.urandom(64)
    sigs = [PopSchemeMPL.sign(sk, msg) for sk in sks]
    print(f"sign,1,{timeit(lambda: PopSchemeMPL.sign(sks[0], msg), REPS):.4f}", flush=True)
    print(f"verify,1,{timeit(lambda: PopSchemeMPL.verify(pks[0], msg, sigs[0]), REPS):.4f}", flush=True)
    for n in (32, 64, 128):
        q = n - (n - 1) // 3
        agg = PopSchemeMPL.aggregate(sigs[:q])
        assert PopSchemeMPL.fast_aggregate_verify(pks[:q], msg, agg)
        print(f"aggregate,{q},{timeit(lambda: PopSchemeMPL.aggregate(sigs[:q]), REPS):.4f}", flush=True)
        print(f"fast_aggregate_verify,{q},{timeit(lambda: PopSchemeMPL.fast_aggregate_verify(pks[:q], msg, agg), REPS):.4f}", flush=True)


if __name__ == "__main__":
    main()
