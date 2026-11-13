import unittest
import numpy as np
from pathlib import Path
from audit_v3c_stage22 import load_source,objective_and_gradient

class Stage22Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prod=load_source(Path(__file__).parent/'v3c_stage2_reference.py')
    def test_production_feature_dimensions(self):
        h={0:np.array([[0,1,0],[1,1,0.]],float),10:np.array([[1,1,0],[1,0,0.]],float)}
        p,x=self.prod.make_features(h,10,20)
        self.assertEqual([x[k].shape[1] for k in ('M1_shared','M2_history','M3_local_context')],[2,5,8])
    def test_gradient_finite_difference(self):
        h={0:np.array([[0,1,0],[1,1,0.]],float),10:np.array([[1,1,0],[1,0,0.]],float)}
        p,x=self.prod.make_features(h,10,20)
        d=[{'p0':p,'features':x,'y':np.array([[1,1,0],[0,0,1]],float)}]
        w=np.array([0.03,-0.07]);v,g=objective_and_gradient(self.prod,d,'M1_shared',w,0.1)
        for j in range(2):
            step=np.zeros(2);step[j]=1e-5
            vp,_=objective_and_gradient(self.prod,d,'M1_shared',w+step,0.1)
            vm,_=objective_and_gradient(self.prod,d,'M1_shared',w-step,0.1)
            self.assertAlmostEqual(g[j],(vp-vm)/2e-5,places=6)
    def test_future_history_rejected(self):
        with self.assertRaises(ValueError):
            self.prod.make_features({20:np.zeros((2,3))},10,30)
if __name__=='__main__':unittest.main()
