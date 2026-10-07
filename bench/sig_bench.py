"""Client-signature verification cost: secp256k1 ECDSA via libsecp256k1 (coincurve).

Ethereum-style transactions carry a recoverable secp256k1 signature; a node
that acknowledges a vertex checks every transaction in it. Measures plain
verification and public-key recovery (ecrecover).
Usage: python bench/sig_bench.py > results/bench_sig.csv
"""
import os
import time

from coincurve import PrivateKey, PublicKey

N = 5000


def main():
    print("op,rep,us_per_op", flush=True)
    keys = [PrivateKey() for _ in range(64)]
    msgs = [os.urandom(32) for _ in range(N)]
    sigs = [keys[i % 64].sign(m, hasher=None) for i, m in enumerate(msgs)]
    rsigs = [keys[i % 64].sign_recoverable(m, hasher=None) for i, m in enumerate(msgs)]
    pubs = [keys[i % 64].public_key for i in range(N)]
    for rep in range(5):
        t = time.perf_counter()
        for s, m, p in zip(sigs, msgs, pubs):
            assert p.verify(s, m, hasher=None)
        print(f"verify,{rep},{(time.perf_counter() - t) / N * 1e6:.2f}", flush=True)
        t = time.perf_counter()
        for s, m in zip(rsigs, msgs):
            PublicKey.from_signature_and_message(s, m, hasher=None)
        print(f"recover,{rep},{(time.perf_counter() - t) / N * 1e6:.2f}", flush=True)


if __name__ == "__main__":
    main()
