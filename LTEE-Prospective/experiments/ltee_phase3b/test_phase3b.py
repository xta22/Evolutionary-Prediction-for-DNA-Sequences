import unittest
import numpy as np
from scipy.linalg import expm
from phase3b import generator, transition, make_data, Model, objective, fit, evaluate

class Phase3BTests(unittest.TestCase):
    def test_generator(self):
        q = generator(np.zeros((3, 6)))
        self.assertTrue(np.allclose(q.sum(axis=-1), 0))
        self.assertTrue(np.allclose(-np.diagonal(q, axis1=-2, axis2=-1).mean(axis=-1), 1))

    def test_exponential(self):
        q = generator(np.arange(6) * .1)
        for t in [0, .1, .7, 2.]:
            self.assertTrue(np.allclose(transition(q, t), expm(q*t), atol=1e-9))

    def test_probabilities(self):
        q = generator(np.zeros((5, 6)))
        p = transition(q, np.arange(5) * .1)
        self.assertTrue(np.all(p >= 0))
        self.assertTrue(np.allclose(p.sum(-1), 1))

    def test_synthetic_and_model(self):
        x, origin, data, _ = make_data(42, 'linear', 3, 40, 3, 60, 20, 30)
        m = Model(3, 3, 'M3_linear')
        pred = m.predict(data['test'], x, origin)
        self.assertEqual(pred.shape, (30, 4))
        self.assertTrue(np.isfinite(objective(m.theta, m, data['train'], x, origin, .003)))

    def test_fit_smoke(self):
        x, origin, data, _ = make_data(42, 'homogeneous', 2, 20, 3, 30, 15, 20)
        m = Model(2, 3, 'M1')
        result = fit(m, data['train'], data['validation'], x, origin, 3, .003)
        self.assertTrue(np.isfinite(result['best_validation_nll']))
        self.assertTrue(np.isfinite(evaluate(m, data['test'], x, origin)['nll']))

if __name__ == '__main__':
    unittest.main()
