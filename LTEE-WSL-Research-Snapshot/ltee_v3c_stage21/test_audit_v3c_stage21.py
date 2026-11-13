import unittest, tempfile
from pathlib import Path
import pandas as pd
from audit_v3c_stage21 import metric_comparison,inspect_source,optional_table
class AuditTests(unittest.TestCase):
 def test_paired(self):
  d=pd.DataFrame([{'scenario':'x','seed':1,'fold_id':'f','cutoff':10,'target_generation':20,'model':m,'nll':v} for m,v in [('M1_shared',0.5),('M2_history',0.4)]])
  r=metric_comparison(d);self.assertAlmostEqual(r[0]['mean_nll_delta'],-0.1)
 def test_duplicate_rejected(self):
  d=pd.DataFrame([{'scenario':'x','model':'M1_shared','nll':.5}]*2)
  with self.assertRaises(ValueError):metric_comparison(d)
 def test_source(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'x.py';p.write_text('def predict(x):\n return x\n')
   self.assertEqual(inspect_source(p)['functions'][0]['name'],'predict')
 def test_optional(self):self.assertEqual(optional_table(None,'features')['status'],'NOT_AVAILABLE')
if __name__=='__main__':unittest.main()
