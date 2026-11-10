#!/usr/bin/env python3
"""Synthetic correctness checks; no empirical LTEE data or fitting."""
import unittest
import numpy as np
from ctmc_baselines import make_q,transition,identity_probs,ctmc_probs,full_nucleotide_log_loss

class TestM0M1(unittest.TestCase):
    def setUp(self):
        self.Q=make_q(np.array([[0,1,2,1],[1,0,1,2],[2,1,0,1],[1,2,1,0]],float)*1e-5)
    def test_zero_rate(self):
        self.assertTrue(np.allclose(transition(np.zeros((4,4)),100000),np.eye(4)))
    def test_zero_time(self):
        self.assertTrue(np.allclose(transition(self.Q,0),np.eye(4)))
    def test_stochastic(self):
        for t in [1,1000,10000,50000]:
            p=transition(self.Q,t)
            self.assertTrue((p>=0).all())
            self.assertTrue(np.allclose(p.sum(axis=1),1))
    def test_semigroup(self):
        a,b=3000,7000
        self.assertTrue(np.allclose(transition(self.Q,a)@transition(self.Q,b),transition(self.Q,a+b),atol=1e-12))
    def test_time_sensitivity(self):
        self.assertGreater(1-transition(self.Q,10000)[0,0],1-transition(self.Q,1000)[0,0])
    def test_identity(self):
        self.assertTrue(np.array_equal(identity_probs('ACGT'),np.eye(4)))
    def test_ctmc_probabilities(self):
        self.assertTrue(np.allclose(ctmc_probs('ACGT',self.Q,1000),transition(self.Q,1000)))
    def test_logloss(self):
        self.assertAlmostEqual(full_nucleotide_log_loss(identity_probs('ACGT'),'ACGT'),0)
    def test_reject_invalid_q(self):
        with self.assertRaises(ValueError): transition(np.ones((4,4)),100)
    def test_recover_known_rate_from_simulated_transitions(self):
        rng=np.random.default_rng(42)
        q=make_q(np.full((4,4),2e-5))
        t=5000
        true_p=transition(q,t)
        n=50000
        starts=rng.integers(0,4,size=n)
        ends=np.array([rng.choice(4,p=true_p[a]) for a in starts])
        estimated=np.zeros((4,4))
        for a in range(4):
            x=ends[starts==a]
            estimated[a]=np.bincount(x,minlength=4)/len(x)
        self.assertLess(np.max(np.abs(estimated-true_p)),0.015)

if __name__=='__main__':unittest.main(verbosity=2)
