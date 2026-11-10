"""Deterministic checks of chart formulas and finite-horizon reporting.

Run from this directory with ``python -m unittest -v test_qc_monitoring.py``
or from the workspace with
``python -m unittest -v qc_python_starter.test_qc_monitoring``.
No simulations, model training demos, or population-ARL assertions are used.
"""

import math
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy import stats

if __package__:
    from . import qc_monitoring as qc
else:
    import qc_monitoring as qc


class FeatureTests(unittest.TestCase):
    def test_features_keep_fixed_signed_node_separate_from_max_abs(self):
        u = np.array([
            [[1, -2, -3], [4, 5, 9], [-6, 7, -8]],
            [[-1, 2, 4], [6, -7, -5], [8, 9, -10]],
        ], dtype=float)
        maximum, node, profile = qc.extract_features(u, critical_node=0)
        assert_array_equal(maximum, [9, 10])
        assert_array_equal(node, [[1, -2, -3], [-1, 2, 4]])
        assert_array_equal(profile, [[-3, 9, -8], [4, -5, -10]])
        node[:] = 0
        profile[:] = 0
        self.assertEqual(u[0, 0, 2], -3, "Returned features must not alias U")

    def test_features_reject_invalid_shape_node_and_nonfinite_data(self):
        for u in (np.zeros((2, 3)), np.zeros((2, 3, 2)),
                  np.zeros((2, 3, 4)), np.empty((0, 3, 3)),
                  np.full((2, 3, 3), np.nan)):
            with self.subTest(shape=u.shape), self.assertRaises(ValueError):
                qc.extract_features(u, critical_node=0)
        for node in (-1, 3, 1.5, True):
            with self.subTest(node=node), self.assertRaises(ValueError):
                qc.extract_features(np.zeros((2, 3, 3)), critical_node=node)

    def test_subgroups_are_consecutive_and_reject_incomplete_groups(self):
        assert_array_equal(qc.subgroup(np.arange(6), n=3), [[0, 1, 2], [3, 4, 5]])
        for values, n in ((np.arange(5), 3), (np.arange(6), 1),
                          (np.arange(6), True), (np.arange(6), 2.0),
                          (np.zeros((2, 3)), 3), ([], 2)):
            with self.subTest(values=values, n=n), self.assertRaises(ValueError):
                qc.subgroup(values, n)


class XbarSTests(unittest.TestCase):
    def test_c4_closed_forms_and_validation(self):
        for n, expected in ((2, math.sqrt(2 / math.pi)),
                            (3, math.sqrt(math.pi) / 2),
                            (5, 3 * math.sqrt(2 * math.pi) / 8)):
            with self.subTest(n=n):
                self.assertAlmostEqual(qc.c4(n), expected, places=14)
        for n in (0, 1, 2.0, True):
            with self.subTest(n=n), self.assertRaises(ValueError):
                qc.c4(n)

    def test_fit_uses_mean_sample_s_over_c4_not_pooled_variance(self):
        # Sample SDs are 1 and 2, despite very different subgroup means.
        groups = np.array([[0, 1, 2], [8, 10, 12]], dtype=float)
        chart = qc.XbarS.fit(groups.ravel(), n=3)
        self.assertEqual(chart.n, 3)
        self.assertEqual(chart.mu, 5.5)
        self.assertAlmostEqual(chart.sigma, 3 / math.sqrt(math.pi))
        means, sample_s = chart.statistics(groups)
        assert_allclose(means, [1, 10])
        assert_allclose(sample_s, [1, 2])
        limits = chart.moment_limits(width=2)
        self.assertAlmostEqual(limits["s_cl"], 1.5)
        self.assertAlmostEqual(limits["s_ucl"], 1.5 + 2 * chart.sigma * math.sqrt(1 - math.pi / 4))
        self.assertAlmostEqual(limits["xbar_ucl"], 5.5 + 2 * chart.sigma / math.sqrt(3))
        for groups in ([0, 1, 2], [[0, 1]], [[0, np.nan, 2]]):
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                chart.statistics(groups)

    def test_probability_limits_have_requested_normal_and_chi2_tails(self):
        chart = qc.XbarS(n=5, mu=7, sigma=2)
        limits = chart.probability_limits(alpha_x=0.1, alpha_s=0.2)
        se = 2 / math.sqrt(5)
        self.assertAlmostEqual(stats.norm.cdf((limits["xbar_lcl"] - 7) / se), 0.05)
        self.assertAlmostEqual(stats.norm.sf((limits["xbar_ucl"] - 7) / se), 0.05)
        self.assertAlmostEqual(stats.chi2.cdf(4 * (limits["s_lcl"] / 2)**2, 4), 0.1)
        self.assertAlmostEqual(stats.chi2.sf(4 * (limits["s_ucl"] / 2)**2, 4), 0.1)
        self.assertAlmostEqual(limits["s_cl"], 2 * qc.c4(5))
        for alpha in (0, 1, -0.1, np.nan):
            for kwargs in ({"alpha_x": alpha, "alpha_s": 0.2},
                           {"alpha_x": 0.1, "alpha_s": alpha}):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    chart.probability_limits(**kwargs)

    def test_joint_score_is_or_of_moment_limit_violations(self):
        chart = qc.XbarS(n=3, mu=0, sigma=1)
        # In control, mean-only, large spread, zero spread, negative mean.
        groups = np.array([[-1, 0, 1], [3, 4, 5], [-4, 0, 4],
                           [0, 0, 0], [-5, -4, -3]], dtype=float)
        mean, sd = chart.statistics(groups)
        for width in (0.5, 2.0, 4.0):
            with self.subTest(width=width):
                limits = chart.moment_limits(width)
                violations = ((mean < limits["xbar_lcl"]) | (mean > limits["xbar_ucl"])
                              | (sd < limits["s_lcl"]) | (sd > limits["s_ucl"]))
                assert_array_equal(chart.joint_score(groups) > width, violations)
                batched = groups.reshape(1, 5, 3)
                assert_allclose(chart.joint_score(batched), chart.joint_score(groups)[None, :])
        self.assertEqual(chart.moment_limits(4)["s_lcl"], 0)


class SequentialChartTests(unittest.TestCase):
    def test_cusum_hand_recurrence_runs_resets_and_strict_alarms(self):
        z = np.array([1, 1, -2, -1, -1, 3], dtype=float)
        table = qc.cusum_table(10 + 2 * z, mu=10, sigma=2, k=0.5, h=1)
        assert_array_equal(table["sheet"], np.arange(1, 7))
        assert_allclose(table["z"], z)
        assert_allclose(table["upper_increment"], z - 0.5)
        assert_allclose(table["lower_increment"], -z - 0.5)
        assert_allclose(table["upper"], [0.5, 1, 0, 0, 0, 2.5])
        assert_allclose(table["lower"], [0, 0, 1.5, 2, 2.5, 0])
        assert_array_equal(table["upper_run"], [1, 2, 0, 0, 0, 1])
        assert_array_equal(table["lower_run"], [0, 0, 1, 2, 3, 0])
        assert_array_equal(table["upper_alarm"], [False, False, False, False, False, True])
        assert_array_equal(table["lower_alarm"], [False, False, True, True, True, False])
        lower_tie = qc.cusum_table([-1, -1, -1], 0, 1, k=0.5, h=1)
        assert_array_equal(lower_tie["lower_alarm"], [False, False, True])

    def test_cusum_paths_start_independently_and_calls_do_not_retain_state(self):
        paths = np.array([[1, 1, -2], [-1, -1, 2]], dtype=float)
        expected = [[0.5, 1, 0], [0, 0, 1.5]]
        for _ in range(2):
            assert_allclose(qc.cusum_score_paths(paths, 0, 1), expected)
        assert_allclose(qc.cusum_score_paths([[1, -1, 2]], 0, 1, k=0), [[1, 0, 2]])
        for kwargs in ({"k": -1}, {"k": np.nan}, {"sigma": 0}):
            args = dict(mu=0, sigma=1)
            args.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                qc.cusum_score_paths(paths, **args)

    def test_ewma_hand_recurrence_and_startup_variance(self):
        # Standardized states: 1, 1/2, -3/4. Startup variances:
        # 1/4, 5/16, 21/64 (not the asymptotic variance 1/3).
        paths = np.array([[14, 10, 6], [6, 10, 14]], dtype=float)
        states = np.array([1, 0.5, -0.75])
        factors = np.sqrt([1 / 4, 5 / 16, 21 / 64])
        expected = states / factors
        assert_allclose(qc.ewma_score_paths(paths, 10, 2, lam=0.5), [expected, -expected])
        table = qc.ewma_table(paths[0], 10, 2, lam=0.5, L=2)
        assert_allclose(table["ewma"], [12, 11, 8.5])
        assert_allclose(table["startup_sd"], 2 * factors)
        assert_allclose(table["ucl"], 10 + 4 * factors)
        assert_allclose(table["lcl"], 10 - 4 * factors)
        assert_allclose(table["center"], 10)
        assert_allclose(table["score"], expected)
        assert_array_equal(table["upper_alarm"], [False, False, False])

    def test_ewma_lam_one_is_signed_z_with_strict_upper_alarm(self):
        table = qc.ewma_table([3, 3.5, -4], 0, 1, lam=1, L=3)
        assert_allclose(table["ewma"], [3, 3.5, -4])
        assert_allclose(table["score"], [3, 3.5, -4])
        assert_allclose(table["startup_sd"], 1)
        assert_array_equal(table["upper_alarm"], [False, True, False])
        for lam in (0, -0.1, 1.1, np.nan):
            with self.subTest(lam=lam), self.assertRaises(ValueError):
                qc.ewma_score_paths([[1, 2]], 0, 1, lam=lam)

    def test_ma_window_one_and_growing_startup(self):
        y = np.array([12, 8, 14, 6], dtype=float)
        assert_allclose(qc.ma_score(y, 10, 2, window=1, two_sided=False), [1, -1, 2, -2])
        assert_allclose(qc.ma_score(y, 10, 2, window=1), [1, 1, 2, 2])
        expected = np.array([1, 0, 2 / math.sqrt(3), -1 / math.sqrt(3)])
        assert_allclose(qc.ma_score(y, 10, 2, window=3, two_sided=False), expected)
        assert_allclose(qc.ma_score(y, 10, 2, window=3), np.abs(expected))
        assert_allclose(qc.ma_score(y, 10, 2, window=10, two_sided=False),
                        [1, 0, 2 / math.sqrt(3), 0])
        for window in (0, -1, 1.5, True):
            with self.subTest(window=window), self.assertRaises(ValueError):
                qc.ma_score(y, 10, 2, window=window)

    def test_mewma_lam_one_equals_hotelling_and_startup_is_normalized(self):
        chart = qc.Hotelling(np.array([1, -2]), np.array([[4, 2], [2, 2]]))
        # Cholesky factor [[2, 0], [1, 1]] maps these known whitened rows.
        z = np.array([[1, 2], [-1, 0], [0, -2]], dtype=float)
        x = z @ np.array([[2, 1], [0, 1]]) + [1, -2]
        assert_allclose(qc.mewma_score(x, chart, lam=1), chart.score(x))
        # lam=1/2 states: (1/2,1), (-1/4,1/2), (-1/8,-3/4).
        assert_allclose(qc.mewma_score(x, chart, lam=0.5), [5, 1, 37 / 21])
        for lam in (0, 1.1, np.nan):
            with self.subTest(lam=lam), self.assertRaises(ValueError):
                qc.mewma_score(x, chart, lam)


class HotellingTests(unittest.TestCase):
    def test_known_covariance_scores_and_chi_square_limit(self):
        mean = np.array([1., -2.])
        covariance = np.array([[4., 2.], [2., 2.]])
        chart = qc.Hotelling(mean, covariance)
        x = np.array([[1, -2], [3, 1], [-1, -3]], dtype=float)
        assert_allclose(chart.whiten(x), [[0, 0], [1, 2], [-1, 0]])
        assert_allclose(chart.score(x), [0, 5, 1])
        # Chi-square(2) survival function is exp(-x/2).
        for phase in (1, 2):
            self.assertAlmostEqual(chart.analytic_limit(0.05, phase), -2 * math.log(0.05))
        mean[:] = 99
        covariance[:] = 0
        assert_allclose(chart.score(x), [0, 5, 1])
        with self.assertRaises(ValueError):
            chart.score([[1, 2, 3]])

    def test_phase_one_beta_limit_and_degrees_of_freedom(self):
        # p=2, m=5 gives Beta(1,1), whose quantile is 1-alpha.
        chart = qc.Hotelling(np.zeros(2), np.eye(2), reference_count=5)
        self.assertAlmostEqual(chart.analytic_limit(0.1, phase=1), (16 / 5) * 0.9)
        boundary = qc.Hotelling(np.zeros(2), np.eye(2), reference_count=3)
        with self.assertRaises(ValueError):
            boundary.analytic_limit(0.1, phase=1)
        self.assertTrue(np.isfinite(boundary.analytic_limit(0.1, phase=2)))
        for kwargs in ({"alpha": 0}, {"alpha": 1}, {"phase": 3}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                chart.analytic_limit(**kwargs)

    def test_fitted_univariate_phase_two_is_predictive_t_squared(self):
        baseline = np.array([[0], [1], [2], [3], [4]], dtype=float)
        chart = qc.Hotelling.fit(baseline)
        assert_allclose(chart.mean, [2])
        assert_allclose(chart.covariance, [[2.5]])  # sample, not population variance
        self.assertEqual(chart.reference_count, 5)
        alpha = 0.1
        t_quantile = stats.t.ppf(1 - alpha / 2, df=4)
        self.assertAlmostEqual(chart.analytic_limit(alpha), (1 + 1 / 5) * t_quantile**2)
        future = 2 + math.sqrt(2.5 * (1 + 1 / 5)) * t_quantile
        self.assertAlmostEqual(chart.score([[future]])[0], chart.analytic_limit(alpha))

    def test_covariance_rejection_and_insufficient_baselines(self):
        for covariance in (np.eye(3), [[1, 0.2], [0, 1]],
                           [[1, 1], [1, 1]], [[1, 2], [2, 1]],
                           [[1, 0], [0, 1e-14]], [[np.nan, 0], [0, 1]]):
            with self.subTest(covariance=covariance), self.assertRaises(ValueError):
                qc.Hotelling(np.zeros(2), covariance)
        for baseline in ([[0, 0], [1, 1]], [[0, 0], [1, 1], [2, 2]]):
            with self.subTest(baseline=baseline), self.assertRaises(ValueError):
                qc.Hotelling.fit(baseline)
        with self.assertRaises(ValueError):
            qc.Hotelling(np.zeros(2), np.eye(2), reference_count=2)


class PCATests(unittest.TestCase):
    @staticmethod
    def baseline():
        # Orthogonal, unequal sample eigenvalues: 18/5, 8/5, 2/5.
        return np.array([[3, 0, 0, 7], [-3, 0, 0, 7],
                         [0, 2, 0, 7], [0, -2, 0, 7],
                         [0, 0, 1, 7], [0, 0, -1, 7]], dtype=float)

    def test_pca_sample_eigen_scaling_and_orthogonal_q_residual(self):
        baseline = self.baseline()
        chart = qc.PCAMonitor.fit(baseline, r=1, standardize=False)
        self.assertEqual(chart.r, 1)
        assert_allclose(chart.pca.explained_variance_, [18 / 5])
        x = np.array([[3, 2, -1, 7], [0, -4, 2, 7]], dtype=float)
        t2, q = chart.scores(x)
        assert_allclose(t2, [2.5, 0], atol=1e-14)
        assert_allclose(q, [5, 20])
        z = chart.transform(x)
        residual = z - chart.pca.inverse_transform(chart.pca.transform(z))
        assert_allclose(residual @ chart.pca.components_.T, 0, atol=1e-14)
        assert_allclose(np.sum(residual**2, axis=1), q)
        assert_allclose(np.sum(chart.whitened_scores(x)**2, axis=1), t2)
        self.assertAlmostEqual(chart.q_scale, 10 / 6)
        white = chart.whitened_scores(baseline)
        assert_allclose(white.var(axis=0, ddof=1), [1])

    def test_pca_preprocessing_and_constant_mask_remain_frozen(self):
        baseline = self.baseline()
        chart = qc.PCAMonitor.fit(baseline, r=1, standardize=True)
        assert_array_equal(chart.mask, [True, True, True, False])
        assert_allclose(chart.mean, [0, 0, 0, 7])
        assert_allclose(chart.scale, np.sqrt([18 / 5, 8 / 5, 2 / 5, 1]))
        frozen = [a.copy() for a in (chart.mean, chart.scale, chart.mask,
                                    chart.pca.components_, chart.pca.explained_variance_, chart.pca.mean_)]
        q_scale = chart.q_scale
        x = np.array([[6, 4, 2, 1000], [-3, -2, -1, -1000]], dtype=float)
        expected = x[:, :3] / np.sqrt([18 / 5, 8 / 5, 2 / 5])
        assert_allclose(chart.transform(x), expected)
        # Neither a varying formerly constant coordinate nor batch composition
        # may change training-time preprocessing or scores for a given row.
        unchanged_coordinate = x.copy()
        unchanged_coordinate[:, 3] = 7
        assert_allclose(chart.scores(x), chart.scores(unchanged_coordinate))
        together = chart.joint_score(x)
        separately = np.concatenate([chart.joint_score(row[None, :]) for row in x])
        assert_allclose(together, separately)
        baseline[:] = 99
        for actual, expected in zip((chart.mean, chart.scale, chart.mask,
                                     chart.pca.components_, chart.pca.explained_variance_, chart.pca.mean_), frozen):
            assert_array_equal(actual, expected)
        self.assertEqual(chart.q_scale, q_scale)

    def test_pca_joint_score_is_or_and_rejects_undefined_models(self):
        chart = qc.PCAMonitor.fit(self.baseline(), r=2, standardize=False)
        x = np.array([[0, 0, 0, 7], [9, 0, 0, 7], [0, 0, 3, 7], [6, 4, 2, 7]], dtype=float)
        t2, q = chart.scores(x)
        score = chart.joint_score(x)
        assert_allclose(chart.pca.explained_variance_, [18 / 5, 8 / 5])
        self.assertAlmostEqual(chart.q_scale, 1 / 3)
        assert_allclose(score, np.maximum(t2 / 2, q / (1 / 3)))
        for limit in (0, 1, 5, float(score[1])):
            with self.subTest(limit=limit):
                assert_array_equal(score > limit, (t2 / chart.r > limit) | (q / chart.q_scale > limit))
        for data, r in ((self.baseline(), 3), (self.baseline()[:2], 1),
                        (np.ones((4, 3)), 1),
                        (np.array([[-1, -2], [0, 0], [1, 2]]), 1)):
            with self.subTest(data=data, r=r), self.assertRaises(ValueError):
                qc.PCAMonitor.fit(data, r=r)
        with self.assertRaises(ValueError):
            chart.transform(np.zeros((2, 3)))


class LossTests(unittest.TestCase):
    def test_loss_bounds_clipping_and_mixed_feasibility(self):
        assert_allclose(qc.loss_from_reward([0, 0.25, 1]), [1, 0.75, 0])
        d = [0, 1.5, 3, 6]
        feasibility = [1, 0, 0.5, 0.25]
        assert_allclose(qc.bounded_loss(d), [0, 0.5, 1, 1])
        mixed = qc.bounded_loss(d, feasibility_loss=feasibility, weight=0.25)
        assert_allclose(mixed, [0.75, 0.125, 0.625, 0.4375])
        self.assertTrue(np.all((mixed >= 0) & (mixed <= 1)))
        assert_allclose(qc.bounded_loss(d, feasibility_loss=feasibility, weight=0), feasibility)
        assert_allclose(qc.bounded_loss(d, feasibility_loss=feasibility, weight=1), [0, 0.5, 1, 1])
        assert_allclose(qc.bounded_loss([1, 2], deformation_scale=2), [0.5, 1])

    def test_losses_reject_invalid_bounds_shapes_and_missing_feasibility(self):
        for reward in ([-0.01], [1.01], [np.nan], [np.inf], [], [[0.5]]):
            with self.subTest(reward=reward), self.assertRaises(ValueError):
                qc.loss_from_reward(reward)
        for kwargs in ({"d": [-1]}, {"d": [np.inf]}, {"d": [[1]]},
                       {"deformation_scale": 0}, {"deformation_scale": np.nan},
                       {"weight": -0.1}, {"weight": 1.1}, {"weight": np.nan},
                       {"weight": 0.5}, {"feasibility_loss": [0, 1]},
                       {"feasibility_loss": [-0.1]}, {"feasibility_loss": [1.1]},
                       {"feasibility_loss": [np.nan]}, {"feasibility_loss": [[0]]}):
            args = {"d": [1]}
            args.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                qc.bounded_loss(**args)


class FiniteHorizonTests(unittest.TestCase):
    def test_finite_rank_threshold_rounding_ties_and_too_small_sample(self):
        # n=5, alpha=.4 -> ceil(6*.6)=4, not a percentile interpolation.
        scores = np.array([8., 1., 3., 3., 2.])
        original = scores.copy()
        self.assertEqual(qc.upper_limit(scores, alpha=0.4), 3)
        assert_array_equal(scores, original)
        assert_array_equal(qc.first_alarm(scores[:, None], 3), [1, 2, 2, 2, 2])
        self.assertEqual(qc.upper_limit([2, 2, 2], alpha=0.25), 2)
        self.assertEqual(qc.upper_limit([9], alpha=0.5), 9)
        for values, alpha in (([1, 2, 3], 0.2), ([9], 0.49),
                              ([1, 2], 0), ([1, 2], 1), ([], 0.5),
                              ([1, np.nan], 0.5), ([[1, 2]], 0.5)):
            with self.subTest(values=values, alpha=alpha), self.assertRaises(ValueError):
                qc.upper_limit(values, alpha)

    def test_first_alarm_strict_ties_and_horizon_censor_sentinel(self):
        paths = [[2, 2, 2], [0, 2, 3], [3, 0, 4], [0, 3, 0]]
        assert_array_equal(qc.first_alarm(paths, 2), [4, 3, 1, 2])
        for paths, limit in (([[1]], np.inf), ([[1]], np.nan),
                             ([1, 2], 1), ([[1, np.nan]], 1), (np.empty((2, 0)), 1)):
            with self.subTest(paths=paths, limit=limit), self.assertRaises(ValueError):
                qc.first_alarm(paths, limit)

    def test_run_length_sheet_units_censoring_and_sample_mcse(self):
        # Observed updates 1, 2, 4 become 5, 10, 20 sheets.
        paths = [[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 0, 2]]
        result = qc.run_length_summary(paths, 1, unit_sheets=5)
        self.assertEqual(result["replications"], 3)
        self.assertEqual(result["signals"], 3)
        self.assertEqual(result["unit_sheets"], 5)
        self.assertEqual(result["horizon_sheets"], 20)
        self.assertEqual(result["censored_fraction"], 0)
        self.assertAlmostEqual(result["restricted_mean_sheets"], 35 / 3)
        self.assertAlmostEqual(result["arl_sheets"], 35 / 3)
        self.assertAlmostEqual(result["restricted_mean_mcse"], math.sqrt(175) / 3)
        censored = qc.run_length_summary(paths[:2] + [[0, 0, 0, 0]], 1, unit_sheets=5)
        self.assertEqual(censored["signals"], 2)
        self.assertAlmostEqual(censored["censored_fraction"], 1 / 3)
        self.assertIsNone(censored["arl_sheets"])
        for key in ("restricted_mean_sheets", "restricted_mean_mcse"):
            self.assertEqual(censored[key], result[key])
        single = qc.run_length_summary([[0, 2]], 1)
        self.assertEqual(single["arl_sheets"], 2)
        self.assertIsNone(single["restricted_mean_mcse"])
        all_censored = qc.run_length_summary([[0, 0], [1, 1]], 1)
        self.assertIsNone(all_censored["arl_sheets"])
        self.assertEqual(all_censored["restricted_mean_mcse"], 0)
        for units in (0, True, 1.5):
            with self.subTest(units=units), self.assertRaises(ValueError):
                qc.run_length_summary(paths, 1, unit_sheets=units)

    def test_fault_delay_conditions_on_survivors_and_counts_censoring(self):
        paths = [[2, 0, 0, 0, 0], [0, 2, 0, 0, 0],  # early, including change boundary
                 [0, 0, 2, 0, 0], [0, 0, 0, 0, 2], [0, 0, 0, 0, 0]]
        result = qc.fault_delay_summary(paths, 1, change_index=2, unit_sheets=5)
        self.assertEqual(result["replications"], 5)
        self.assertEqual(result["survivors"], 3)
        self.assertEqual(result["change_index_updates"], 2)
        self.assertEqual(result["post_change_horizon_sheets"], 15)
        self.assertAlmostEqual(result["early_alarm_fraction"], 2 / 5)
        self.assertAlmostEqual(result["detection_probability"], 2 / 3)
        self.assertAlmostEqual(result["censored_fraction"], 1 / 3)
        # Survivor delays are 5, 15, and capped 15, not averages over all paths.
        self.assertAlmostEqual(result["restricted_delay_sheets"], 35 / 3)
        self.assertAlmostEqual(result["delay_mcse"], 10 / 3)
        self.assertIsNone(result["mean_delay_sheets"])

    def test_fault_delay_no_survivors_single_survivor_and_change_at_start(self):
        none = qc.fault_delay_summary([[2, 0, 0], [0, 2, 0]], 1, change_index=2)
        self.assertEqual(none["survivors"], 0)
        self.assertEqual(none["early_alarm_fraction"], 1)
        for key in ("detection_probability", "censored_fraction", "restricted_delay_sheets",
                    "delay_mcse", "mean_delay_sheets"):
            self.assertIsNone(none[key], key)
        one = qc.fault_delay_summary([[2, 0, 0], [0, 0, 2]], 1, change_index=1, unit_sheets=5)
        self.assertEqual(one["survivors"], 1)
        self.assertEqual(one["mean_delay_sheets"], 10)
        self.assertEqual(one["detection_probability"], 1)
        self.assertIsNone(one["delay_mcse"])
        paths = [[0, 2, 0], [0, 0, 0]]
        delay = qc.fault_delay_summary(paths, 1, change_index=0, unit_sheets=2)
        run = qc.run_length_summary(paths, 1, unit_sheets=2)
        self.assertEqual(delay["early_alarm_fraction"], 0)
        self.assertEqual(delay["restricted_delay_sheets"], run["restricted_mean_sheets"])
        self.assertEqual(delay["delay_mcse"], run["restricted_mean_mcse"])
        self.assertIsNone(delay["mean_delay_sheets"])
        censored = qc.fault_delay_summary([[0, 0, 0]], 1, change_index=1)
        self.assertEqual(censored["detection_probability"], 0)
        self.assertEqual(censored["censored_fraction"], 1)
        self.assertEqual(censored["restricted_delay_sheets"], 2)
        self.assertIsNone(censored["mean_delay_sheets"])
        for change in (-1, 3, True, 1.5):
            with self.subTest(change=change), self.assertRaises(ValueError):
                qc.fault_delay_summary(paths, 1, change_index=change)

    def test_calibration_monotonicity_exact_and_nearest_attainable_targets(self):
        # Identical increasing paths have attainable means 1,2,3,4,5 updates.
        paths = np.tile(np.arange(1., 6.), (3, 1))
        original = paths.copy()
        previous_tau = np.zeros(3)
        previous_mean = 0
        for limit in (0, 1, 1.5, 2, 3, 4, 5):
            tau = qc.first_alarm(paths, limit)
            result = qc.run_length_summary(paths, limit, unit_sheets=5)
            self.assertTrue(np.all(tau >= previous_tau))
            self.assertGreaterEqual(result["restricted_mean_sheets"], previous_mean)
            previous_tau, previous_mean = tau, result["restricted_mean_sheets"]
        limits = []
        for target in (10, 15, 20):
            limit, result = qc.calibrate_run_length(paths, target_sheets=target, unit_sheets=5)
            limits.append(limit)
            self.assertEqual(result["target_sheets"], target)
            self.assertEqual(result["arl_sheets"], target)
            self.assertEqual(result["absolute_target_error"], 0)
            self.assertEqual(result["limit"], limit)
            checked = qc.run_length_summary(paths, limit, unit_sheets=5)
            for key, value in checked.items():
                self.assertEqual(result[key], value, key)
        self.assertEqual(limits, sorted(limits))
        # 13 sheets is unattainable on this step function; 15 is the nearest.
        _, approximate = qc.calibrate_run_length(paths, target_sheets=13, unit_sheets=5)
        self.assertEqual(approximate["restricted_mean_sheets"], 15)
        self.assertEqual(approximate["absolute_target_error"], 2)
        assert_array_equal(paths, original)

    def test_calibration_horizon_validation_and_excessive_censor_rejection(self):
        paths = np.tile(np.arange(1., 6.), (2, 1))
        for target in (0, 5, 25, 26, np.nan, np.inf):
            with self.subTest(target=target), self.assertRaises(ValueError):
                qc.calibrate_run_length(paths, target_sheets=target, unit_sheets=5)
        for kwargs in ({"max_censoring": -0.1}, {"max_censoring": 1.1},
                       {"max_censoring": np.nan}, {"iterations": 0},
                       {"iterations": True}, {"unit_sheets": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                qc.calibrate_run_length(paths, target_sheets=3, **kwargs)
        # Constant paths can only alarm at update 1 or be censored at H=4.
        # For target 3.5 the nearest restricted mean is 4, but ARL is unknown.
        constant = np.ones((3, 4))
        with self.assertRaisesRegex(ValueError, "censor"):
            qc.calibrate_run_length(constant, target_sheets=3.5)
        _, allowed = qc.calibrate_run_length(constant, target_sheets=3.5, max_censoring=1)
        self.assertEqual(allowed["restricted_mean_sheets"], 4)
        self.assertEqual(allowed["absolute_target_error"], 0.5)
        self.assertEqual(allowed["censored_fraction"], 1)
        self.assertIsNone(allowed["arl_sheets"])


if __name__ == "__main__":
    unittest.main()