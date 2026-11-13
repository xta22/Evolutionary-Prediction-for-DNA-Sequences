import unittest
import numpy as np
from v3c_stage23 import expand,fit,predict,synthetic_fixture

class Tests(unittest.TestCase):
    def sample(self):
        rng=np.random.default_rng(4)
        x=np.column_stack([np.ones(30),rng.normal(size=(30,7))])
        p0=np.full(30,.25)
        y=rng.binomial(1,.25,size=(8,30))
        return {'features':{'M3_local_context':x},'p0':p0,'y':y}
    def test_feature_shapes(self):
        d=self.sample()
        self.assertEqual([expand(d,k).shape[1] for k in ('shared','history','local','nonlinear')],[2,5,8,17])
    def test_fit_predict(self):
        d=self.sample();r=fit([d],'local',.1)
        p=predict(d,'local',r['coef'])
        self.assertTrue(np.isfinite(p).all());self.assertTrue(((p>0)&(p<1)).all())
    def test_gradients(self):
        d=self.sample();r=fit([d],'shared',.1)
        self.assertTrue(np.isfinite(r['train_final_gradient_norm']))
        self.assertLessEqual(r['train_final_objective'],r['train_initial_objective']+1e-9)
    def test_oracle_fixture(self):
        d=synthetic_fixture(917)
        self.assertLess(d['oracle_hidden_z_nll'],d['observable_noisy_proxy_nll'])
        self.assertLess(d['observable_noisy_proxy_nll'],d['uninformative_noise_nll'])
if __name__=='__main__':unittest.main()
