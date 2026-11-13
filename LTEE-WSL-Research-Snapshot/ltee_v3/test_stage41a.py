import unittest
from pathlib import Path
import numpy as np
from stage41a import load_production, intervention, matched_shuffled_distribution

class MatchedAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prod=load_production(Path(__file__).with_name('stage4.py'))
        cls.mat=np.array([[0,0],[0,1],[1,0],[1,1],[1,1],[0,1]],dtype=np.int8)
        cls.pairs=[(0,1)]
        cls.q=cls.prod.frequencies(cls.mat,cls.pairs)
        cls.row={'latest':cls.q,'previous':cls.q,'latest_matrix':cls.mat,'pairs':cls.pairs,
                 'horizon':10,'history_interval':10,'truth':cls.q}

    def test_shuffling_preserves_marginals(self):
        q,emp=matched_shuffled_distribution(self.prod,self.row,0.5,np.random.default_rng(1))
        self.assertTrue(np.allclose(self.prod.independent(emp),self.prod.independent(self.q)))
        self.assertTrue(np.allclose(self.prod.independent(q),self.prod.independent(self.q),atol=1e-7))

    def test_zero_alpha_returns_empirical(self):
        original=self.prod.predict(self.q,self.q,'M5_shrunk',alpha=0)
        self.assertTrue(np.allclose(original,self.q))

    def test_alpha_one_returns_independence(self):
        preds,_=intervention(self.prod,self.row,1.,np.random.default_rng(2))
        self.assertTrue(np.allclose(preds['original_shrunk'],preds['independent']))
        self.assertTrue(np.allclose(preds['shuffled_matched'],preds['independent']))

    def test_score_finite(self):
        preds,_=intervention(self.prod,self.row,.5,np.random.default_rng(3))
        for q in preds.values():
            self.assertTrue(np.isfinite(self.prod.score(q,self.q)['joint_nll']))

if __name__=='__main__':unittest.main()
