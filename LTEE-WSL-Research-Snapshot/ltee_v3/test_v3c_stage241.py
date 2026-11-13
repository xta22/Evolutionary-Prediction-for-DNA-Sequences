import unittest
import numpy as np
from v3c_stage241 import classify, proxy_step, historical_rate_proxy, sampled_population
class Stage241Tests(unittest.TestCase):
    def test_shift_boundaries(self):
        self.assertEqual(classify(30,40,50),'pre_shift')
        self.assertEqual(classify(40,60,50),'crossing_shift')
        self.assertEqual(classify(60,70,50),'post_shift')
    def test_zero_proxy_mutation(self):
        p=np.zeros((10,8),dtype=np.uint8)
        q=proxy_step(p,np.random.default_rng(4),0.)
        self.assertTrue(np.array_equal(p,q))
    def test_history_leakage_rejected(self):
        h=[{'generation':'10','sequence':'000'},{'generation':'20','sequence':'111'}]
        with self.assertRaises(RuntimeError):historical_rate_proxy(h,10,.01,3)
    def test_sampled_bootstrap(self):
        h=[{'generation':'10','sequence':'000'},{'generation':'10','sequence':'111'}]
        a=sampled_population(h,10,20,7)
        self.assertEqual(a.shape,(20,3))
        self.assertTrue(np.isin(a.sum(axis=1),[0,3]).all())
if __name__=='__main__':unittest.main()
