import unittest
from types import SimpleNamespace
import numpy as np
from v3c_stage24 import initialize,step,replay,oracle_predict,evaluate

class Tests(unittest.TestCase):
    def setUp(self):
        self.a=SimpleNamespace(individuals=24,sites=12,initial_frequency=.25,context_sigma=1.,
            rate=.005,heterogeneity_multiplier=3.,recombination_probability=.5,
            shift_generation=3,shift_multiplier=4.,selection_strength=.1,epistasis_strength=.2,
            generations=6,observe_step=2,sample_size=4)
    def test_replay_parity(self):
        rng,pop,c=initialize(42,self.a);obs={}
        for t in range(7):
            if t%2==0:obs[t]=[''.join(map(str,x)) for x in pop[:4]]
            if t<6:pop=step(pop,rng,c,'recombination',t,self.a)
        states,ctx=replay(42,'recombination',self.a,{2,4},obs)
        self.assertEqual(states[2].shape,(24,12))
    def test_replay_detects_mismatch(self):
        rng,pop,c=initialize(42,self.a)
        obs={t:['0'*12]*4 for t in [0,2,4,6]}
        with self.assertRaises(RuntimeError):replay(42,'neutral',self.a,{2},obs)
    def test_oracle_probabilities(self):
        _,pop,c=initialize(2,self.a)
        p=oracle_predict(pop,c,'epistasis',1,3,self.a,2,10)
        self.assertEqual(p.shape,(12,));self.assertTrue(np.all((p>=0)&(p<=1)))
    def test_nll(self):
        y=np.array([[1,0],[0,1]],float)
        self.assertAlmostEqual(evaluate(np.array([.5,.5]),y)['nll'],np.log(2))

if __name__=='__main__':unittest.main()
