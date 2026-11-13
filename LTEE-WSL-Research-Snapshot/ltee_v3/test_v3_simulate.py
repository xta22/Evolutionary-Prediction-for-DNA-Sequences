import unittest
import numpy as np
from v3_simulate import simulate_population,build_folds,audit,context_rates,fitness
class Tests(unittest.TestCase):
    def setUp(self):
        self.f=np.array([0,1,2,3,0,1,2,3,0,1,2,3],dtype=np.uint8)
    def test_context_normalized(self):
        self.assertAlmostEqual(context_rates(self.f).mean(),1)
    def test_selection_off(self):
        h=np.tile(self.f,(10,1));self.assertTrue(np.all(fitness(h,'neutral',4,8)==1))
    def test_repeatability(self):
        a=simulate_population(np.random.default_rng(42),self.f,'neutral','P00',20,10,0.01,[5,10])
        b=simulate_population(np.random.default_rng(42),self.f,'neutral','P00',20,10,0.01,[5,10])
        self.assertEqual(a,b)
    def test_zero_rate_identity(self):
        a,_,_=simulate_population(np.random.default_rng(42),self.f,'neutral','P00',20,10,0,[5,10])
        self.assertTrue(all(r['sequence']==''.join('ACGT'[x] for x in self.f) for r in a))
    def test_chronology(self):
        obs,_,_=simulate_population(np.random.default_rng(42),self.f,'neutral','P00',20,20,0.01,[10,20])
        f,h,t=build_folds(obs,[10],10)
        self.assertEqual(audit(obs,f,h,t)['status'],'PASS_STRUCTURAL_CHECKS_ONLY')
        h[0]['generation']=999
        with self.assertRaises(AssertionError):audit(obs,f,h,t)
if __name__=='__main__':unittest.main()
