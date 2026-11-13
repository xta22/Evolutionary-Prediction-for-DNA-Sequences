import unittest
from pathlib import Path
import numpy as np
from intervene_v3c_trends import load_production, fixture_checks

class ProductionInterventionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        here=Path(__file__).resolve().parent
        source=here/'v3c_forecast.py'
        cls.prod=load_production(source)

    def test_fixture_calls_actual_production(self):
        result=fixture_checks(self.prod)
        self.assertTrue(result['zero_strength_exact_identity'])
        self.assertGreater(result['B2_nonzero_max_difference'],0)
        self.assertGreater(result['B3_nonzero_max_difference'],0)

    def test_zero_strength_identity(self):
        a={0:np.zeros((3,4)),10:np.ones((3,4))}
        p=self.prod.predict(a,10,20,'B1_latest',5,0)
        for model in ['B2_shrunk_trend','B3_site_trend']:
            np.testing.assert_array_equal(p,self.prod.predict(a,10,20,model,5,0))

    def test_future_history_rejected(self):
        a={0:np.zeros((3,4)),20:np.ones((3,4))}
        with self.assertRaises(ValueError):
            self.prod.predict(a,10,30,'B2_shrunk_trend',5,1)

if __name__=='__main__':unittest.main()
