import unittest
import numpy as np
from phase2a import CLASSES, generator, transition, simulate, fit, predict, metrics
class Tests(unittest.TestCase):
    def test_classes(self):self.assertEqual(len(CLASSES),6)
    def test_zero_time(self):np.testing.assert_allclose(transition(-3,np.zeros(6),0),np.eye(4))
    def test_stochastic(self):
        p=transition(-2,np.arange(6)/10,3)
        self.assertTrue(np.all(p>=-1e-12));np.testing.assert_allclose(p.sum(axis=1),1)
    def test_semigroup(self):
        q=np.arange(6)/10
        np.testing.assert_allclose(transition(-2,q,1.5),transition(-2,q,.5)@transition(-2,q,1),atol=1e-12)
    def test_generator(self):
        q=generator(-2,np.zeros(6));np.testing.assert_allclose(q.sum(axis=1),0,atol=1e-12)
    def test_end_to_end(self):
        rng=np.random.default_rng(3);rows,_=simulate(rng,2,12,.08,.5,.3)
        fit1=fit(rows,'M1',2,0,5)
        p=predict(rows,'M1',fit1,2)
        self.assertEqual(p.shape,(24,4));self.assertTrue(np.isfinite(metrics(rows,p)['nucleotide_nll']))
if __name__=='__main__':unittest.main()
