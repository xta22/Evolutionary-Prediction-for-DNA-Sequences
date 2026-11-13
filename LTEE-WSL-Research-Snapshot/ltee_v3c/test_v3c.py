import unittest
import numpy as np
from v3c_forecast import parse_history,predict,score,select_hyperparams

class TestV3C(unittest.TestCase):
    def test_future_rows_filtered(self):
        rows=[dict(generation='0',sequence='000'),dict(generation='10',sequence='111')]
        arr=parse_history(rows,0)
        self.assertEqual(list(arr),[0])
    def test_future_in_history_rejected(self):
        with self.assertRaises(ValueError):predict({20:np.zeros((2,3))},10,30,'B1_latest',1,0)
    def test_persistence(self):
        p=predict({10:np.ones((4,3))},10,20,'B1_latest',1,0)
        self.assertTrue(np.all((p>0.5)&(p<1)))
    def test_probabilities(self):
        arr={0:np.zeros((4,3)),10:np.ones((4,3))}
        for model in ['B0_global','B1_latest','B2_shrunk_trend','B3_site_trend']:
            p=predict(arr,10,20,model,1,1)
            self.assertTrue(np.all((p>0)&(p<1)))
    def test_score(self):
        s=score(np.array([0.5,0.5]),np.array([[0,1],[1,0]]))
        self.assertAlmostEqual(s['nll'],np.log(2))
        self.assertAlmostEqual(s['predicted_allele1_count'],2)
    def test_no_target_in_predict_signature(self):
        import inspect
        self.assertNotIn('target_labels',inspect.signature(predict).parameters)

if __name__=='__main__':unittest.main()
