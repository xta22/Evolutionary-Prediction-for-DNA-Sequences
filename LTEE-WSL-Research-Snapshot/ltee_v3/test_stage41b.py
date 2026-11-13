import unittest
import numpy as np
from stage41b import features,shuffle_joint,predict
class Fake:
    @staticmethod
    def independent(q):
        a=q[:,2]+q[:,3];b=q[:,1]+q[:,3]
        return np.column_stack([(1-a)*(1-b),(1-a)*b,a*(1-b),a*b])
    @staticmethod
    def normalize(q):
        q=np.maximum(q,1e-9);return q/q.sum(axis=1,keepdims=True)
class Tests(unittest.TestCase):
    def setUp(self):
        mat=np.array([[0,0],[0,0],[1,1],[1,1],[0,1],[1,0]],dtype=int)
        q=np.bincount(2*mat[:,0]+mat[:,1],minlength=4)[None,:]/len(mat)
        self.r={'latest_matrix':mat,'pairs':[(0,1)],'latest':q,'previous':q.copy(),'horizon':10}
    def test_features(self):
        self.assertEqual(features(self.r,'history').shape,(1,4))
    def test_shuffled_marginals(self):
        q=shuffle_joint(self.r,12);o=self.r['latest']
        self.assertAlmostEqual(q[0,2]+q[0,3],o[0,2]+o[0,3]);self.assertAlmostEqual(q[0,1]+q[0,3],o[0,1]+o[0,3])
    def test_rho_zero_independence(self):
        q=predict(self.r,np.array([-30.]),'global',Fake())
        self.assertTrue(np.allclose(q,Fake.independent(self.r['latest']),atol=1e-9))
    def test_valid_probability(self):
        q=predict(self.r,np.zeros(4),'history',Fake())
        self.assertTrue(np.all(q>=0));self.assertTrue(np.allclose(q.sum(axis=1),1))
    def test_matched_shuffle(self):
        theta=np.zeros(4)
        q=predict(self.r,theta,'history',Fake(),True,12)
        self.assertTrue(np.allclose(q.sum(axis=1),1))
if __name__=='__main__':unittest.main()
