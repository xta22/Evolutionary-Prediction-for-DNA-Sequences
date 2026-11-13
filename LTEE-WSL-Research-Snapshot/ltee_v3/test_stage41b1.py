import unittest
import numpy as np
from audit_stage41b1 import design_summary,group_trajectories

class Dummy:
    @staticmethod
    def features(r,kind):
        h=r['horizon']/10
        if kind=='horizon':return np.array([[1.,h]])
        return np.array([[1.]])

class AuditTests(unittest.TestCase):
    def test_rank_deficient_constant_horizon(self):
        rows=[{'horizon':10,'scenario':'a','seed':1} for _ in range(3)]
        s=design_summary(rows,Dummy,'horizon')
        self.assertEqual(s['rank'],1)
        self.assertFalse(s['full_column_rank'])
    def test_rank_full_varying_horizon(self):
        rows=[{'horizon':h,'scenario':'a','seed':1} for h in (10,20,30)]
        self.assertEqual(design_summary(rows,Dummy,'horizon')['rank'],2)
    def test_trajectory_groups(self):
        rows=[{'scenario':'a','seed':1},{'scenario':'a','seed':1},{'scenario':'b','seed':1}]
        self.assertEqual(len(group_trajectories(rows)),2)

if __name__=='__main__':unittest.main()
