import unittest
from audit_v3c_trends import audit_metrics,synthetic_fixture
class Tests(unittest.TestCase):
 def test_fixture(self):
  self.assertEqual(synthetic_fixture()['nonzero_trend_prediction'],[0.5,0.5])
 def test_equal_metrics(self):
  rows=[dict(split='test',scenario='neutral',fold_id='1',model=m,nll='0.4',brier='0.2',abs_count_error='1') for m in ('B1_latest','B2_shrunk_trend','B3_site_trend')]
  r=audit_metrics(rows)
  self.assertEqual(r['comparisons']['B2_shrunk_trend_vs_B1']['exact_equal'],1)
 def test_different_metrics(self):
  rows=[dict(split='test',scenario='neutral',fold_id='1',model=m,nll=str(x),brier='0.2',abs_count_error='1') for m,x in [('B1_latest',.4),('B2_shrunk_trend',.3),('B3_site_trend',.4)]]
  r=audit_metrics(rows)
  self.assertEqual(r['comparisons']['B2_shrunk_trend_vs_B1']['exact_equal'],0)
if __name__=='__main__': unittest.main()
