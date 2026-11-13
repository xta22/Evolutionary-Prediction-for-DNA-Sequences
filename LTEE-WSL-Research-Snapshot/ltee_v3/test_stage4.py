import unittest
import numpy as np
from stage4 import frequencies,independent,predict,score,pairs_for_length

class Stage4Tests(unittest.TestCase):
    def test_joint_counts(self):
        x=np.array([[0,0],[0,1],[1,0],[1,1]],dtype=np.int8)
        np.testing.assert_allclose(frequencies(x,[(0,1)]),[[.25]*4])
    def test_independent_preserves_marginals(self):
        q=np.array([[.1,.2,.3,.4]])
        p=independent(q)
        self.assertAlmostEqual(p[0,2]+p[0,3],.7)
        self.assertAlmostEqual(p[0,1]+p[0,3],.6)
    def test_empirical_exact(self):
        q=np.array([[.1,.2,.3,.4]])
        np.testing.assert_allclose(predict(q,None,'M5_empirical'),q)
    def test_shrink_independent(self):
        q=np.array([[.1,.2,.3,.4]])
        np.testing.assert_allclose(predict(q,None,'M5_shrunk',alpha=1),independent(q))
    def test_shuffle_breaks_perfect_linkage(self):
        x=np.array([[0,0]]*50+[[1,1]]*50,dtype=np.int8)
        q=frequencies(x,[(0,1)])
        p=predict(q,None,'M5_shuffled',shuffle_seed=1,latest_matrix=x,pairs=[(0,1)])
        self.assertLess(abs(p[0,3]-.25),.1)
    def test_score_perfect_better(self):
        t=np.array([[.1,.2,.3,.4]])
        self.assertLess(score(t,t)['joint_nll'],score(np.array([[.25]*4]),t)['joint_nll'])
    def test_pair_selection(self):
        self.assertEqual(pairs_for_length(6),[(0,1),(2,3),(4,5)])
if __name__=='__main__':unittest.main()
