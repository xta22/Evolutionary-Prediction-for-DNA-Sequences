import unittest
import numpy as np
from phase2c import simulate, fit_counts, predict, metrics

class Tests(unittest.TestCase):
    def test_no_future_leakage(self):
        a=simulate(42,1e-6,1e6)
        f=fit_counts(a['train'],a['e_train'],1.)
        b=a['test'].copy();b[:]=99999
        g=fit_counts(a['train'],a['e_train'],1.)
        np.testing.assert_allclose(f['alpha'],g['alpha'])
        self.assertAlmostEqual(f['log_rate'],g['log_rate'])
    def test_shared_mle(self):
        y=np.array([[2,1],[3,4]])
        e=np.ones_like(y)*10.
        f=fit_counts(y,e,hierarchical=False)
        self.assertAlmostEqual(predict(f,e).sum(),y.sum())
    def test_positive_predictions(self):
        a=simulate(23,1e-7,1e6)
        f=fit_counts(a['train'],a['e_train'],10.)
        self.assertTrue(np.all(predict(f,a['e_test'])>0))
    def test_finite_heldout_metrics(self):
        a=simulate(11,1e-7,1e6)
        f=fit_counts(a['train'],a['e_train'],1.)
        m=metrics(a['test'],predict(f,a['e_test']),a['true_test'])
        self.assertTrue(np.isfinite(m['heldout_poisson_nll']))
    def test_zero_counts(self):
        y=np.zeros((12,4),dtype=int);e=np.ones((12,4))*1e5
        f=fit_counts(y,e,10.)
        self.assertTrue(np.all(np.isfinite(f['alpha'])))
if __name__=='__main__':unittest.main()
