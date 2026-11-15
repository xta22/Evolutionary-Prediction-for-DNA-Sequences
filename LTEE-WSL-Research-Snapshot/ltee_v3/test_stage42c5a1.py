import unittest
import numpy as np
from stage42c5a1 import probs,pair_counts
class Tests(unittest.TestCase):
 def test_distribution(self):
  c=np.array([[2.,1.,3.,2.],[8.,0.,0.,0.]])
  for alpha in [0,.5,1,5]:
   for rho in [0,.3,1]:
    p=probs(c,alpha,.01,rho)
    self.assertTrue(np.all(p>0))
    np.testing.assert_allclose(p.sum(1),1,atol=1e-10)
 def test_independence(self):
  c=np.array([[2.,1.,3.,2.]])
  p=probs(c,1,0,0)[0];self.assertAlmostEqual(p[3],(5+1)/10*(3+1)/10)
 def test_pair_counts(self):
  pop=np.array([[0,0,1,1],[0,1,1,0]],dtype=np.uint8)
  np.testing.assert_array_equal(pair_counts(pop),[[1,1,0,0],[0,0,1,1]])
if __name__=='__main__':unittest.main()
