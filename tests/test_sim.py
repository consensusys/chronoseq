"""Sanity tests for the timing simulator (sim/)."""

import unittest

import numpy as np

from sim.core import Config
from sim.runner import order_rank, sender_tickets, simulate_dag


class TestSimulator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = Config(duration=4, warmup=1, load_tps=5000, drain=4)
        cls.r = simulate_dag(cls.cfg, seed=7)

    def test_deterministic(self):
        again = simulate_dag(self.cfg, seed=7)
        np.testing.assert_array_equal(self.r.Ep.cut, again.Ep.cut)
        np.testing.assert_array_equal(order_rank(self.r, "cs"), order_rank(again, "cs"))

    def test_cuts_monotone_and_complete(self):
        cut = self.r.Ep.cut
        self.assertTrue((np.diff(cut, axis=0) >= 0).all())
        self.assertTrue((cut[-1] == self.r.L.lane_count).all())
        self.assertTrue((self.r.tx_epoch >= 0).all())

    def test_sequencing_after_submission(self):
        lat = self.r.latency(0.85)
        self.assertTrue(np.isfinite(lat).all())
        self.assertTrue((lat > 0).all())

    def test_per_sender_contiguity(self):
        """Transactions of one sender are contiguous and in nonce order within an epoch."""
        run = self.r
        r = order_rank(run, "cs")
        w = run.w
        for e in np.unique(run.tx_epoch)[:5]:
            idx = np.flatnonzero(run.tx_epoch == e)
            idx = idx[np.argsort(r[idx])]
            senders = w.client[idx]
            change = np.flatnonzero(senders[1:] != senders[:-1])
            blocks = np.split(senders, change + 1)
            self.assertEqual(len({b[0] for b in blocks}), len(blocks))  # contiguous
            for u in np.unique(senders)[:20]:
                n = w.nonce[idx][senders == u]
                self.assertTrue((np.diff(n) > 0).all())

    def test_sender_tickets_depend_on_seed(self):
        t = sender_tickets(self.r)
        e = self.r.tx_epoch
        c = self.r.w.client
        same_client = np.flatnonzero(c == c[0])
        if np.unique(e[same_client]).size > 1:
            self.assertGreater(np.unique(t[same_client]).size, 1)


if __name__ == "__main__":
    unittest.main()
