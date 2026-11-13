import unittest
import numpy as np
from v3b1_audit import fitness_table, interaction_identity, haplotype_run, initial_population, mutation_context, rate_heterogeneity
class TestMechanisms(unittest.TestCase):
    def test_additive(self): self.assertAlmostEqual(interaction_identity(.02,.03,0),0,places=12)
    def test_epistasis(self): self.assertAlmostEqual(interaction_identity(.02,.03,.08),.08,places=12)
    def test_fitness(self): self.assertTrue((fitness_table(.02,.03,.08)>0).all())
    def test_reproducible(self):
        init=initial_population(42,80)
        a=haplotype_run(5,80,10,.02,.02,.08,.5,init)
        b=haplotype_run(5,80,10,.02,.02,.08,.5,init)
        np.testing.assert_allclose(a,b,equal_nan=True)
    def test_no_change(self):
        init=initial_population(42,80)
        t=haplotype_run(5,80,10,0,0,0,0,init,0)
        self.assertTrue(np.all((t[:,1:]>=0) | np.isnan(t[:,1:])))
    def test_context(self): self.assertGreater(mutation_context(11,1000,70,240,1)['context_expected_ratio'],1)
    def test_rates(self): self.assertGreater(rate_heterogeneity(11,12,30,.8)['rate_true_cv'],0)
if __name__=='__main__': unittest.main()
