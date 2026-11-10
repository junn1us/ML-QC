"""Adapter and workflow regressions, separate from the mathematical checks."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal

if __package__:
    from . import demo, qc_monitoring as qc
else:
    import demo
    import qc_monitoring as qc


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.U = demo.synthetic_sheets(200, np.random.default_rng(42))
        cls.new = demo.synthetic_sheets(100, np.random.default_rng(43))
        cls.models = demo.fit_models(cls.U)

    def test_svm_sign_and_all_transforms_remain_frozen(self):
        model = self.models
        before = (model.pca.mean.copy(), model.pca.pca.components_.copy(),
                  model.svm.support_vectors_.copy(), model.loss_mu, model.loss_sigma)
        pool = demo.score_pool(self.new, model)
        assert_allclose(pool["svm"], -model.svm.decision_function(model.pca.whitened_scores(self.new[:, :, 2])))
        demo.score_pool(self.new + 0.1, model)
        for expected, actual in zip(before, (model.pca.mean, model.pca.pca.components_,
                                            model.svm.support_vectors_, model.loss_mu, model.loss_sigma)):
            assert_array_equal(actual, expected)

    def test_actual_rewards_are_consistent_at_every_stage(self):
        rewards = np.linspace(0.1, 0.9, len(self.U))
        models = demo.fit_models(self.U, reward=rewards)
        self.assertAlmostEqual(models.loss_mu, np.mean(1 - rewards))
        new_rewards = np.linspace(0.2, 0.7, len(self.new))
        assert_allclose(demo.score_pool(self.new, models, reward=new_rewards)["loss"], 1 - new_rewards)
        for model, reward in ((models, None), (self.models, new_rewards), (models, new_rewards[:-1])):
            with self.assertRaises(ValueError):
                demo.score_pool(self.new, model, reward=reward)
        with self.assertRaises(ValueError):
            demo.save_trace(Path("unused"), {}, models)

    def test_paired_paths_group_only_within_rows(self):
        indices = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]])
        pool = demo.score_pool(self.new, self.models)
        paths = {name: (p, unit) for name, p, unit in demo.iter_chart_paths(pool, indices, self.models)}
        self.assertEqual(tuple(paths), demo.METHODS)
        assert_allclose(paths["XbarS"][0], self.models.chart.joint_score(pool["d"][indices].reshape(2, 2, 5)))
        self.assertEqual(paths["XbarS"][1], 5)
        assert_allclose(paths["Hotelling"][0], pool["hotelling"][indices])
        for row in range(2):
            assert_allclose(paths["CUSUM"][0][row], qc.cusum_table(pool["loss"][indices[row]], self.models.loss_mu,
                                                                 self.models.loss_sigma)["upper"])
        with self.assertRaises(ValueError):
            list(demo.iter_chart_paths(pool, indices[:, :9], self.models))

    def test_direction_only_fault_does_not_change_uz_based_detectors(self):
        normal = demo.synthetic_sheets(100, np.random.default_rng(3))
        fault = demo.synthetic_sheets(100, np.random.default_rng(3), "Ux_direction", 2)
        assert_array_equal(normal[:, :, 1:], fault[:, :, 1:])
        a, b = demo.score_pool(normal, self.models), demo.score_pool(fault, self.models)
        for key in ("d", "loss", "pca", "pca_q", "pca_t2", "svm"):
            assert_array_equal(a[key], b[key])
        self.assertFalse(np.allclose(a["hotelling"], b["hotelling"]))

    def test_external_contract_grid_rewards_and_duplicate_archives(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / f"split{i}.npz" for i in range(3)]
            for i, path in enumerate(paths):
                np.savez(path, U=self.U + i * .01, reward=np.linspace(.2, .8, len(self.U)),
                         measurement_coordinates=demo.measurement_grid())
            archives, records = demo._load_external(paths)
            self.assertEqual(len(archives), 3)
            self.assertEqual(len({r["sha256"] for r in records}), 3)
            with self.assertRaises(ValueError):
                demo._load_external([paths[0], paths[0], paths[2]])
            np.savez(paths[2], U=self.U, reward=np.linspace(.2, .8, len(self.U)),
                     measurement_coordinates=demo.measurement_grid()[::-1])
            with self.assertRaises(ValueError):
                demo._load_external(paths)
            np.savez(paths[2], U=self.U, measurement_coordinates=demo.measurement_grid())
            with self.assertRaises(ValueError):
                demo._load_external(paths)
            np.savez(paths[2], U=self.U[:, :, :2])
            with self.assertRaises(ValueError):
                demo._load_external(paths)

    def test_seed_streams_are_distinct_and_repeatable(self):
        a, b = demo._seed_plan(123), demo._seed_plan(123)
        self.assertEqual(len({s.spawn_key for s in a.values()}), len(a))
        for key in a:
            assert_array_equal(np.random.default_rng(a[key]).normal(size=5),
                               np.random.default_rng(b[key]).normal(size=5))

    def test_external_run_emits_only_in_control_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [root / f"split{i}.npz" for i in range(3)]
            for i, path in enumerate(paths):
                U = demo.synthetic_sheets(2000, np.random.default_rng(400 + i))
                reward = 1 - np.max(np.abs(U[:, :, 2]), axis=1) / 3
                np.savez(path, U=U, reward=reward, measurement_coordinates=demo.measurement_grid())
            args = demo._parser().parse_args(["--out", str(root / "out"), "--fit-file", str(paths[0]),
                                             "--cal-file", str(paths[1]), "--test-file", str(paths[2]),
                                             "--data-id", "unit-test-toy", "--replications", "16",
                                             "--horizon", "500", "--target-sheets", "20"])
            result = demo.run(args)
            self.assertEqual(result["mode"], "external_IID_bootstrap")
            self.assertEqual(set(p.name for p in (root / "out").iterdir()),
                             {"calibration.csv", "validation.csv", "run_metadata.json"})
            with (root / "out" / "run_metadata.json").open() as handle:
                self.assertIsNone(json.load(handle)["faults"])
            self.assertEqual(len(pd.read_csv(root / "out" / "validation.csv")), 6)
            with self.assertRaises(ValueError):
                demo.run(args)


if __name__ == "__main__":
    unittest.main()