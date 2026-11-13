import unittest
from types import SimpleNamespace
from v3b_simulate import simulate, make_folds, ld_r2

class Tests(unittest.TestCase):
    def cfg(self,**kwargs):
        d=dict(individuals=40,sites=24,generations=20,observe_step=5,horizon=5,sample_size=4,rate=.002,initial_frequency=.25,context_sigma=1.,heterogeneity_multiplier=3.,shift_generation=10,shift_multiplier=4.,recombination_probability=.5,selection_strength=.15,epistasis_strength=.3)
        d.update(kwargs);return SimpleNamespace(**d)
    def test_repeatable(self):
        a=simulate('neutral',11,self.cfg());b=simulate('neutral',11,self.cfg())
        self.assertEqual(a['observations'],b['observations'])
    def test_no_mutation(self):
        a=simulate('neutral',11,self.cfg(rate=0.))
        self.assertEqual(sum(x['mutation_events'] for x in a['events']),0)
    def test_recombination_only_switch(self):
        a=simulate('recombination',11,self.cfg(recombination_probability=1.))
        b=simulate('linkage_only',11,self.cfg(recombination_probability=1.))
        self.assertEqual(sum(x['crossovers'] for x in a['events']),800)
        self.assertEqual(sum(x['crossovers'] for x in b['events']),0)
    def test_regime_rate_expectation(self):
        a=simulate('regime_shift',11,self.cfg())
        self.assertAlmostEqual(a['events'][0]['expected_mutations']*4,a['events'][-1]['expected_mutations'])
    def test_epistasis_switch(self):
        a=simulate('epistasis',11,self.cfg());b=simulate('selection_only',11,self.cfg())
        self.assertEqual(a['truth']['epistasis_strength'],.3)
        self.assertEqual(b['truth']['epistasis_strength'],0)
    def test_fold_temporal(self):
        a=simulate('neutral',11,self.cfg());folds,h,t=make_folds(a['observations'],self.cfg())
        self.assertGreater(len(folds),0)
        self.assertTrue(all(x['generation']<=int(x['fold_id'].split('_')[0]) for x in h))
        self.assertTrue(all(x['generation']==int(x['fold_id'].split('_')[-1]) for x in t))
    def test_ld(self):
        import numpy as np
        p=np.array([[0,0],[0,0],[1,1],[1,1]],dtype='uint8')
        self.assertAlmostEqual(ld_r2(p,0,1),1.)
if __name__=='__main__':unittest.main()
