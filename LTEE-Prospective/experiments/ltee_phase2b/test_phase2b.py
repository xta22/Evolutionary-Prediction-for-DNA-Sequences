import unittest
import numpy as np
from phase2b import recovery, rare_simulate, fit_rare
from phase2a import generator, transition

class Tests(unittest.TestCase):
    def test_generator(self):
        q=generator(np.log(1e-5),np.zeros(6))
        self.assertTrue(np.allclose(q.sum(1),0))
    def test_transition(self):
        p=transition(np.log(0.1),np.zeros(6),1)
        self.assertTrue(np.allclose(p.sum(1),1))
    def test_rare_zero_approx(self):
        _,prob,_=rare_simulate(np.random.default_rng(1),3,10000,1e-10,0,1)
        self.assertTrue(np.all(prob<1e-7))
    def test_rare_fit(self):
        result,_=fit_rare([12,13,11],100000,1,10,np.log(1e-4))
        self.assertTrue(result.success)
    def test_recovery(self):
        self.assertEqual(recovery(None,{},2),{})

if __name__=='__main__':unittest.main()
