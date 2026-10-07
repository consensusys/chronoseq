/* Microbenchmark of ChronoSeq's per-epoch ordering step (Algorithm 2, lines 3-7):
 * de-duplicate by (sender, nonce), compute one SHA-256 ticket per sender,
 * sort by (ticket, sender, nonce), and compute the Merkle root of the result.
 * m transactions from m/2 distinct senders, 1% duplicates. Single-threaded.
 * Build: gcc -O2 -o order_bench order_bench.c -lcrypto
 * Usage: ./order_bench m reps -> CSV: m,rep,ns_per_tx,ns_order_per_tx,senders,kept,check
 */
#include <openssl/evp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct { uint8_t sender[20]; uint64_t nonce; uint8_t hash[32]; uint8_t ticket[32]; } tx_t;

static uint64_t rng = 88172645463325252ULL;
static uint64_t xs(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }

static int cmp_sn(const void *a, const void *b) {
  const tx_t *x = *(tx_t *const *)a, *y = *(tx_t *const *)b;
  int c = memcmp(x->sender, y->sender, 20);
  if (c) return c;
  if (x->nonce != y->nonce) return x->nonce < y->nonce ? -1 : 1;
  return memcmp(x->hash, y->hash, 32);
}
static int cmp_tk(const void *a, const void *b) {
  const tx_t *x = *(tx_t *const *)a, *y = *(tx_t *const *)b;
  int c = memcmp(x->ticket, y->ticket, 32);
  if (c) return c;
  c = memcmp(x->sender, y->sender, 20);
  if (c) return c;
  return x->nonce < y->nonce ? -1 : (x->nonce > y->nonce);
}
static EVP_MD_CTX *ctx; static const EVP_MD *md;
static void sha(const uint8_t *in, size_t n, uint8_t *out) {
  EVP_DigestInit_ex(ctx, md, NULL); EVP_DigestUpdate(ctx, in, n); EVP_DigestFinal_ex(ctx, out, NULL);
}
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

int main(int argc, char **argv) {
  size_t m = argc > 1 ? strtoull(argv[1], 0, 10) : 100000;
  int reps = argc > 2 ? atoi(argv[2]) : 5;
  size_t U = m / 2 + 1;                      /* distinct senders */
  tx_t *tx = malloc(m * sizeof(tx_t));
  tx_t **p = malloc(m * sizeof(tx_t *));
  uint8_t (*lv)[32] = malloc(m * 32);
  uint8_t seed[32];
  ctx = EVP_MD_CTX_new(); md = EVP_sha256();
  for (int i = 0; i < 32; i++) seed[i] = (uint8_t)xs();
  for (int r = 0; r < reps; r++) {
    for (size_t i = 0; i < m; i++) {         /* fresh workload, 1% duplicates */
      uint64_t s = xs() % U;
      memset(tx[i].sender, 0, 20); memcpy(tx[i].sender, &s, 8);
      tx[i].nonce = xs() % 1000000;
      if (i && xs() % 100 == 0) tx[i] = tx[i - 1];
      else for (int k = 0; k < 4; k++) { uint64_t v = xs(); memcpy(tx[i].hash + 8 * k, &v, 8); }
      p[i] = &tx[i];
    }
    double t0 = now();
    qsort(p, m, sizeof(tx_t *), cmp_sn);     /* group by sender, nonce order */
    size_t k = 0, senders = 0;
    uint8_t buf[6 + 32 + 20];
    memcpy(buf, "Ticket", 6); memcpy(buf + 6, seed, 32);
    for (size_t i = 0; i < m; i++) {
      if (k && memcmp(p[i]->sender, p[k - 1]->sender, 20) == 0) {
        if (p[i]->nonce == p[k - 1]->nonce) continue;          /* dedup (sender, nonce) */
        memcpy(p[i]->ticket, p[k - 1]->ticket, 32);
      } else {                                                  /* new sender: one ticket */
        memcpy(buf + 38, p[i]->sender, 20);
        sha(buf, sizeof buf, p[i]->ticket);
        senders++;
      }
      p[k++] = p[i];
    }
    qsort(p, k, sizeof(tx_t *), cmp_tk);     /* order by (ticket, sender, nonce) */
    double t1 = now();
    for (size_t i = 0; i < k; i++) memcpy(lv[i], p[i]->hash, 32);
    size_t w = k;                             /* Merkle root over the ordered list */
    while (w > 1) {
      size_t o = 0;
      for (size_t i = 0; i < w; i += 2) {
        if (i + 1 < w) sha(lv[i], 64, lv[o]); else memcpy(lv[o], lv[i], 32);
        o++;
      }
      w = o;
    }
    double dt = now() - t0;
    printf("%zu,%d,%.1f,%.1f,%zu,%zu,%02x\n", m, r, 1e9 * dt / m, 1e9 * (t1 - t0) / m, senders, k, lv[0][0]);
  }
  return 0;
}
