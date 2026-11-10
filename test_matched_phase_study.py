import unittest
import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import binom

if __package__:
    from .matched_phase_study import binomial_summary, generate_profile, score, PCAMonitor
else:
    from matched_phase_study import binomial_summary, generate_profile, score, PCAMonitor


class MatchedStudyTests(unittest.TestCase):
    def test_exact_probability_and_reciprocal_intervals(self):
        s = binomial_summary(10, 1000)
        self.assertAlmostEqual(binom.sf(9, 1000, s['p_lo']), .025)
        self.assertAlmostEqual(binom.cdf(10, 1000, s['p_hi']), .025)
        self.assertEqual(s['arl_hat'], 100)
        self.assertAlmostEqual(s['arl_lo'], 1/s['p_hi'])
        self.assertAlmostEqual(s['arl_hi'], 1/s['p_lo'])
        self.assertAlmostEqual(s['detection_by_h'], 1-.99**50)

    def test_zero_and_all_events(self):
        s = binomial_summary(0, 100)
        self.assertIsNone(s['arl_hat'])
        self.assertIsNone(s['arl_hi'])
        self.assertEqual(s['p_lo'], 0)
        s = binomial_summary(100, 100)
        self.assertEqual(s['p_hi'], 1)
        self.assertEqual(s['arl_lo'], 1)
        self.assertEqual(s['detection_by_h'], 1)
        with self.assertRaises(ValueError):
            binomial_summary(2, 1)

    def test_fault_applied_to_same_physical_profile(self):
        a = generate_profile(30, np.random.default_rng(42))
        b = generate_profile(30, np.random.default_rng(42), 'broad_Uz_mean', 2)
        assert_allclose(b-a, .3, atol=1e-14)
        c = generate_profile(30, np.random.default_rng(42), 'local_Uz_distortion', 4)
        self.assertTrue(np.all(c-a >= 0))
        self.assertLessEqual((c-a).max(), .72)

    def test_scores_freeze_pca_and_use_sample_sd(self):
        fit = generate_profile(200, np.random.default_rng(1))
        model = PCAMonitor.fit(fit)
        d = np.max(abs(fit), axis=1)
        components = model.pca.components_.copy()
        scores = score(fit, model, d.mean(), d.std(ddof=1))
        assert_allclose(scores[:, 1], abs((d-d.mean())/d.std(ddof=1)))
        assert_allclose(scores[:, 0], model.joint_score(fit))
        score(fit + 1, model, d.mean(), d.std(ddof=1))
        assert_allclose(components, model.pca.components_)


if __name__ == '__main__':
    unittest.main()