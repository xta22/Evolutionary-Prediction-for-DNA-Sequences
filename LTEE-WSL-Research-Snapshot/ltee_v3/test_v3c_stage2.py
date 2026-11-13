import unittest
import numpy as np
from v3c_stage2 import make_features, parse_history, fit_model, predict, metrics

class Stage2Tests(unittest.TestCase):
    def setUp(self):
        self.h={0:np.array([[0,1,0,1],[1,1,0,0]],dtype=float),
                10:np.array([[1,1,0,1],[1,0,0,1]],dtype=float)}
    def test_future_history_rejected(self):
        with self.assertRaisesRegex(ValueError,'LEAKAGE'):
            parse_history([{'generation':'20','sequence':'0101'}],10)
    def test_target_rejected(self):
        with self.assertRaises(ValueError): make_features(self.h,10,10)
    def test_features_do_not_depend_on_target(self):
        a=make_features(self.h,10,20)
        b=make_features({k:v.copy() for k,v in self.h.items()},10,20)
        np.testing.assert_array_equal(a[0],b[0])
        np.testing.assert_array_equal(a[1]['M3_local_context'],b[1]['M3_local_context'])
    def test_horizon_changes_features(self):
        a=make_features(self.h,10,20)[1]['M3_local_context']
        b=make_features(self.h,10,30)[1]['M3_local_context']
        self.assertGreater(np.max(np.abs(a-b)),0)
    def test_fit_probabilities(self):
        p0,features=make_features(self.h,10,20)
        d={'p0':p0,'features':features,'y':np.array([[1,1,0,1],[0,1,1,1]],float)}
        fit=fit_model([d],'M3_local_context',0.1)
        p=predict(d,'M3_local_context',fit)
        self.assertTrue(np.all((p>0)&(p<1)))
        self.assertTrue(np.isfinite(metrics(p,d['y'])['nll']))
    def test_training_changes_when_labels_change(self):
        p0,features=make_features(self.h,10,20)
        a={'p0':p0,'features':features,'y':np.ones((5,4))}
        b={'p0':p0,'features':features,'y':np.zeros((5,4))}
        fa=fit_model([a],'M1_shared',0.001)['coef']
        fb=fit_model([b],'M1_shared',0.001)['coef']
        self.assertGreater(np.linalg.norm(fa-fb),0.1)

if __name__=='__main__': unittest.main()
