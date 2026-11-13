import unittest, tempfile, csv
from pathlib import Path
import numpy as np
from v3c_stage3 import make_rows,design,predictions,fit,shuffled

class Stage3Tests(unittest.TestCase):
    def fixture(self, root, bad=False):
        d=Path(root)/'neutral'/'seed_11';d.mkdir(parents=True)
        def write(name,fields,rows):
            with (d/name).open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
        write('folds.csv',['fold_id','cutoff','target_generation'],[{'fold_id':'10_to_20','cutoff':10,'target_generation':20}])
        write('historical_inputs.csv',['fold_id','generation','sequence'],[
            {'fold_id':'10_to_20','generation':0,'sequence':'0000'},
            {'fold_id':'10_to_20','generation':10 if not bad else 30,'sequence':'1100'}])
        write('evaluation_targets_RESTRICTED.csv',['fold_id','generation','sequence'],[
            {'fold_id':'10_to_20','generation':20,'sequence':'1110'}])
    def test_temporal_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.fixture(tmp,bad=True)
            with self.assertRaises(ValueError):make_rows(Path(tmp),[11])
    def test_shapes_probabilities(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.fixture(tmp)
            rows=make_rows(Path(tmp),[11]);d=design(rows,10,0.1)
            self.assertEqual(d['X'].shape,(4,4))
            for name,dim in [('M1_shared',2),('M4_static',8),('M4_latent',11)]:
                p=predictions(np.zeros(dim),d,name)
                self.assertTrue(np.all((p>0)&(p<1)))
    def test_shuffled_no_labels_changed(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.fixture(tmp)
            rows=make_rows(Path(tmp),[11]);s=shuffled(rows,42)
            np.testing.assert_array_equal(rows[0]['y'],s[0]['y'])
    def test_optimizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.fixture(tmp)
            d=design(make_rows(Path(tmp),[11]),10,0.1)
            theta,info=fit(d,'M1_shared',0.1)
            self.assertEqual(theta.shape,(2,))
            self.assertTrue(np.isfinite(theta).all())
if __name__=='__main__':unittest.main()
