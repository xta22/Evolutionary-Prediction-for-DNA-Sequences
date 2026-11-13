import unittest
import numpy as np
from stage32 import normalize,nll_from_gate,fit_gate

class Stage32Tests(unittest.TestCase):
    def test_train_only_normalization(self):
        a=np.array([[1,1,2],[1,3,4]],float);b=np.array([[1,100,200]],float)
        x,y,_=normalize(a,b)
        self.assertAlmostEqual(float(x[:,1].mean()),0)
        self.assertGreater(float(y[0,1]),10)
    def test_constant_gate(self):
        arr=(np.array([.2,.4]),np.array([.8,.6]),np.array([0.,1.]),np.ones(2),[1,1])
        v,_=nll_from_gate(np.array([.5,.5]),arr)
        self.assertTrue(np.isfinite(v))
    def test_fit_gate(self):
        F=np.ones((2,1));arr=(np.array([.1,.1]),np.array([.9,.9]),np.array([1.,1.]),np.ones(2),[1,1])
        w,info=fit_gate(F,arr,[0],.01)
        self.assertTrue(info['success'])
        self.assertGreater(w[0],0)
if __name__=='__main__':unittest.main()
