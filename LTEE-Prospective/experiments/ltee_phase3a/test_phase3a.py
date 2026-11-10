import unittest
import numpy as np
from phase3a import make_data, features, probabilities, fit

class Phase3ATests(unittest.TestCase):
    def test_probability_bounds(self):
        F=np.zeros((5,0));p=probabilities(np.array([-3.,0.,0.]),F,2,1.)
        self.assertTrue(np.all((p>=0)&(p<=1)))
    def test_zero_horizon(self):
        p=probabilities(np.array([-3.,0.,0.]),np.zeros((5,0)),2,0.)
        self.assertTrue(np.all(p==0))
    def test_time_monotonic(self):
        t=np.array([-3.,0.,0.]);F=np.zeros((5,0))
        self.assertTrue(np.all(probabilities(t,F,2,2.)>=probabilities(t,F,2,1.)))
    def test_null_context(self):
        X,_,rate,_,_=make_data(42,'null',n_sites=100)
        self.assertTrue(np.allclose(rate,rate[:,[0]]))
    def test_fit(self):
        X,_,_,ys,_=make_data(11,'linear',n_sites=250)
        theta,info=fit(ys[0],features(X,'linear'),maxiter=50)
        self.assertTrue(np.isfinite(theta).all())
        self.assertTrue(np.isfinite(info['objective']))
    def test_future_independent(self):
        X,_,_,ys,_=make_data(42,'linear',n_sites=300)
        self.assertFalse(np.array_equal(ys[0],ys[2]))

if __name__=='__main__':unittest.main()
