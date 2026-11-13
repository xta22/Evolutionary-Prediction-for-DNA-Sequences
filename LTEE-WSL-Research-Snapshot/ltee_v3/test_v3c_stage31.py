import unittest
import numpy as np
from scipy.special import expit
from audit_v3c_stage31 import nll, evaluate

class FakeProduction:
    def design(self,rows,hs,vs):
        r=rows[0];L=r['L'];base=np.full(L,.4)
        X=np.column_stack([np.ones(L),np.ones(L),np.zeros(L),np.zeros(L)])
        G=np.column_stack([np.ones(L),np.full(L,r['volatility']/vs),np.full(L,r['volatility']/vs)])
        return {'X':X,'G':G,'base':base,'y':r['y'],'n':np.full(L,r['n'])}
    def predictions(self,theta,d,model):
        from scipy.special import logit
        z=logit(d['base']);a=expit(z+d['X']@theta[:4]);b=expit(z+d['X']@theta[4:8]);g=expit(d['G']@theta[8:11]);return (1-g)*a+g*b

class Tests(unittest.TestCase):
    def test_nll(self):
        self.assertAlmostEqual(nll(np.array([.5]),np.array([1.]),np.array([1.])),np.log(2))
    def test_permutation_detects_gate_dependency(self):
        rows=[{'scenario':'s','seed':42,'fold_id':str(i),'cutoff':i,'target_generation':i+1,'volatility':v,'L':2,'n':4,'y':np.array([float(i%2)]*2)} for i,v in enumerate([0.,1.,0.,1.])]
        theta=np.array([0.,0.,0.,0.,2.,0.,0.,0.,0.,5.,0.])
        report,df=evaluate(FakeProduction(),rows,theta,1.,1.,123,20)
        self.assertEqual(len(df),4)
        self.assertGreater(report['expert_separation_mean_abs'],0.1)
        self.assertGreater(report['permutation_max_prediction_change_mean'],0.)
    def test_equal_experts_gate_irrelevant(self):
        rows=[{'scenario':'s','seed':42,'fold_id':str(i),'cutoff':i,'target_generation':i+1,'volatility':float(i),'L':2,'n':2,'y':np.array([0.,1.])} for i in range(3)]
        theta=np.array([0.,0.,0.,0.,0.,0.,0.,0.,1.,4.,0.])
        report,_=evaluate(FakeProduction(),rows,1.*theta,1.,1.,123,10)
        self.assertAlmostEqual(report['permutation_delta_nll_mean'],0.,places=12)

if __name__=='__main__':unittest.main()
