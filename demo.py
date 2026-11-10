"""Six frozen-baseline charts; synthetic illustration or external IID bootstrap.

Run as ``python qc_python_starter/demo.py --out ...`` or
``python -m qc_python_starter.demo --out ...``. External archives must contain
finite signed U with shape (sheets, measurement_sites, 3), in millimetres,
and may contain reward and measurement_coordinates. Never fabricate XYZ from
a scalar surrogate. All three external archives must use the same schema.

Calibration targets an empirical *restricted* first-alarm mean, not a proven
population ARL. Subgroups contain consecutive sheets, never spatial nodes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM

if __package__:
    from . import qc_monitoring as qc
else:
    import qc_monitoring as qc


METHODS = ("XbarS", "Hotelling", "PCA", "OneClassSVM", "CUSUM", "EWMA")
FAULTS = ("broad_Uz_mean", "Ux_direction", "local_Uz_distortion")
COEFFICIENTS = (0.5, 1.0, 2.0)
OUTPUTS = (
    "calibration.csv", "validation.csv", "fault_comparison.csv",
    "sheet_trace.csv", "subgroup_trace.csv", "cusum_table.csv", "ewma_table.csv",
    "example_displacements.npz", "control_charts_demo.png", "run_metadata.json",
)


@dataclass
class Models:
    """Training-only estimators and moments; only ``limits`` is filled later."""

    chart: qc.XbarS
    hotelling: qc.Hotelling
    pca: qc.PCAMonitor
    svm: OneClassSVM
    loss_mu: float
    loss_sigma: float
    critical_node: int
    deformation_scale: float
    n: int
    r: int
    grid_size: int
    fit_size: int
    uses_reward: bool
    k: float = 0.5
    lam: float = 0.1
    limits: dict[str, float] = field(default_factory=dict)


def _positive_integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _displacements(U, critical_node=44, grid_size=None):
    U = np.asarray(U, dtype=float)
    if U.ndim != 3 or U.shape[0] == 0 or U.shape[2] != 3 or not np.isfinite(U).all():
        raise ValueError("U must be nonempty finite signed XYZ, shape (N, G, 3)")
    node = _positive_integer(critical_node, "critical_node", 0)
    if node >= U.shape[1]:
        raise ValueError("critical_node is outside the measurement grid")
    if grid_size is not None and U.shape[1] != grid_size:
        raise ValueError("Measurement-site count differs from the frozen baseline")
    return U


def _loss(d, reward, deformation_scale):
    if reward is None:
        return qc.bounded_loss(d, deformation_scale=deformation_scale)
    loss = qc.loss_from_reward(reward)
    if loss.shape != d.shape:
        raise ValueError("reward must have exactly one value per sheet")
    return loss


def fit_models(U, n=5, r=3, critical_node=44, reward=None, deformation_scale=3.0):
    """Fit once on baseline sheets. Reward, if supplied, defines loss = 1-reward."""
    n = _positive_integer(n, "n", 2)
    r = _positive_integer(r, "r")
    U = _displacements(U, critical_node)
    if len(U) % n:
        raise ValueError("Baseline sheet count must be divisible by n")
    if not np.isfinite(deformation_scale) or deformation_scale <= 0:
        raise ValueError("deformation_scale must be finite and positive")
    d, xyz, profile = qc.extract_features(U, critical_node)
    loss = _loss(d, reward, deformation_scale)
    mu, sigma = float(loss.mean()), float(loss.std(ddof=1))
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Training loss must have positive finite sample SD")
    chart = qc.XbarS.fit(d, n=n)
    hotelling = qc.Hotelling.fit(xyz)
    pca = qc.PCAMonitor.fit(profile, r=r, standardize=True)
    svm = OneClassSVM(kernel="rbf", nu=0.05, gamma=1.0 / r)
    svm.fit(pca.whitened_scores(profile))
    return Models(chart, hotelling, pca, svm, mu, sigma, int(critical_node),
                  float(deformation_scale), n, r, U.shape[1], len(U), reward is not None)


def score_pool(U, models, reward=None):
    """Compute each sheet's frozen features/scores once, before resampling.

    Both introducing and dropping reward after fitting are errors. SVM scores
    are negative decision_function values, so all six monitors alarm upward.
    """
    U = _displacements(U, models.critical_node, models.grid_size)
    if (reward is not None) != models.uses_reward:
        raise ValueError("Reward must be supplied consistently with baseline fitting")
    d, xyz, profile = qc.extract_features(U, models.critical_node)
    t2, q = models.pca.scores(profile)
    return {"d": d, "loss": _loss(d, reward, models.deformation_scale),
            "hotelling": models.hotelling.score(xyz), "pca_t2": t2, "pca_q": q,
            "pca": np.maximum(t2 / models.r, q / models.pca.q_scale),
            "svm": -models.svm.decision_function(models.pca.whitened_scores(profile)).ravel()}


def iter_chart_paths(pool, indices, models):
    """Yield (method, scores[B, updates], sheets_per_update), one method at a time.

    ``indices`` is a shared (B, H) sheet-index array (or one length-H trace).
    Grouping is exclusively along its time axis, with no row-boundary mixing.
    CUSUM and EWMA start at zero on every complete trajectory, without resets.
    """
    indices = np.asarray(indices)
    if indices.ndim == 1:
        indices = indices[None, :]
    if indices.ndim != 2 or not indices.size or indices.dtype.kind not in "iu":
        raise ValueError("indices must be a nonempty integer array of shape (B, H)")
    B, H = indices.shape
    if H % models.n:
        raise ValueError("Trajectory horizon must be divisible by subgroup size n")
    size = len(pool["d"])
    for key in ("d", "loss", "hotelling", "pca", "svm"):
        values = np.asarray(pool[key])
        if values.shape != (size,) or not np.isfinite(values).all():
            raise ValueError(f"Invalid score pool column: {key}")
    if indices.min() < 0 or indices.max() >= size:
        raise ValueError("Trajectory index outside the score pool")
    yield "XbarS", models.chart.joint_score(pool["d"][indices].reshape(B, H // models.n, models.n)), models.n
    yield "Hotelling", pool["hotelling"][indices], 1
    yield "PCA", pool["pca"][indices], 1
    yield "OneClassSVM", pool["svm"][indices], 1
    yield "CUSUM", qc.cusum_score_paths(pool["loss"][indices], models.loss_mu, models.loss_sigma, models.k), 1
    yield "EWMA", qc.ewma_score_paths(pool["loss"][indices], models.loss_mu, models.loss_sigma, models.lam), 1


def measurement_grid():
    """Fixed 10x10 measurement sites, normalized planar coordinates, not fixtures."""
    x, y = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10), indexing="xy")
    return np.column_stack((x.ravel(), y.ravel()))


def synthetic_sheets(size, rng, fault=None, coefficient=0.0):
    """Toy signed fields: independent XYZ latent draws and full-rank noise.

    Three smooth Uz modes plus residual noise make a nondegenerate PCA example.
    Fault coefficients are arbitrary amplitude multipliers, NOT standardized
    shifts, calibrated mechanical interventions, or evidence about real physics.
    """
    size = _positive_integer(size, "size")
    if fault is not None and fault not in FAULTS:
        raise ValueError(f"Unknown toy fault: {fault}")
    if not np.isfinite(coefficient) or coefficient < 0:
        raise ValueError("coefficient must be finite and nonnegative")
    xy = measurement_grid()
    x, y = xy.T
    modes = np.column_stack((np.ones(len(x)), x / np.sqrt(np.mean(x*x)), y / np.sqrt(np.mean(y*y))))
    U = np.empty((size, len(x), 3))
    for component, amplitudes in enumerate(((0.12, 0.08, 0.06), (0.10, 0.07, 0.09), (0.16, 0.12, 0.10))):
        coefficients = rng.normal(size=(size, 3)) * amplitudes
        U[:, :, component] = coefficients @ modes.T + rng.normal(0, 0.035, (size, len(x)))
    U[:, :, 2] += 0.45 + 0.08 * np.cos(np.pi * x) * np.cos(np.pi * y)
    if fault == "broad_Uz_mean":
        U[:, :, 2] += coefficient * 0.15
    elif fault == "Ux_direction":
        U[:, :, 0] += coefficient * 0.15
    elif fault == "local_Uz_distortion":
        bump = np.exp(-((x - 0.45)**2 + (y + 0.35)**2) / (2 * 0.17**2))
        U[:, :, 2] += coefficient * 0.18 * bump
    return U


def _seed_plan(seed):
    root = np.random.SeedSequence(seed)
    fit, cal, validation, faults, trace = root.spawn(5)
    cal_pool, cal_paths = cal.spawn(2)
    val_pool, val_paths = validation.spawn(2)
    streams = {"fit": fit, "cal_pool": cal_pool, "cal_paths": cal_paths,
               "validation_pool": val_pool, "validation_paths": val_paths}
    for i, scenario in enumerate(faults.spawn(9)):
        streams[f"fault_{i}_pool"], streams[f"fault_{i}_paths"] = scenario.spawn(2)
    streams["trace_pre"], streams["trace_post"] = trace.spawn(2)
    return streams


def _indices(size, args, stream):
    return np.random.default_rng(stream).integers(size, size=(args.replications, args.horizon))


def _calibrate(pool, indices, models, target):
    rows = []
    for method, paths, unit in iter_chart_paths(pool, indices, models):
        limit, summary = qc.calibrate_run_length(paths, target_sheets=target, unit_sheets=unit)
        models.limits[method] = float(limit)
        rows.append({"method": method, **summary})
        del paths
    return pd.DataFrame(rows)


def _validate(pool, indices, models):
    rows = []
    for method, paths, unit in iter_chart_paths(pool, indices, models):
        rows.append({"method": method, "limit": models.limits[method],
                     **qc.run_length_summary(paths, models.limits[method], unit)})
        del paths
    return pd.DataFrame(rows)


def _fault_experiments(args, streams, models):
    rows = []
    scenario = 0
    for family in FAULTS:
        for coefficient in COEFFICIENTS:
            U = synthetic_sheets(args.pool_size, np.random.default_rng(streams[f"fault_{scenario}_pool"]),
                                 family, coefficient)
            pool = score_pool(U, models)
            del U
            indices = _indices(args.pool_size, args, streams[f"fault_{scenario}_paths"])
            for method, paths, unit in iter_chart_paths(pool, indices, models):
                rows.append({"fault_family": family, "toy_coefficient": coefficient,
                             "method": method, "unit_sheets": unit, "limit": models.limits[method],
                             **qc.fault_delay_summary(paths, models.limits[method], change_index=0,
                                                     unit_sheets=unit)})
                del paths
            scenario += 1
    return pd.DataFrame(rows)


def save_trace(out, streams, models):
    """Save a synthetic deformation-only trace; not an actual-reward adapter."""
    if models.uses_reward:
        raise ValueError("Synthetic trace cannot represent a model fitted to actual terminal rewards")
    U = np.concatenate((synthetic_sheets(200, np.random.default_rng(streams["trace_pre"])),
                        synthetic_sheets(300, np.random.default_rng(streams["trace_post"]),
                                         "broad_Uz_mean", 1.0)))
    pool = score_pool(U, models)
    sheet = pd.DataFrame({"sheet": np.arange(1, 501), "post_change": np.arange(500) >= 200,
                          "maxabs_Uz_mm": pool["d"], "loss": pool["loss"],
                          "illustrative_reward": 1 - pool["loss"],
                          "hotelling_T2": pool["hotelling"], "pca_T2": pool["pca_t2"],
                          "pca_Q": pool["pca_q"], "pca_joint": pool["pca"], "svm_score": pool["svm"]})
    groups = qc.subgroup(pool["d"], models.n)
    xbar, s = models.chart.statistics(groups)
    subgroup = pd.DataFrame({"group": np.arange(1, len(groups) + 1),
                             "sheet_end": np.arange(models.n, 501, models.n),
                             "xbar_mm": xbar, "s_mm": s,
                             "joint_score": models.chart.joint_score(groups)})
    limits = models.chart.moment_limits(models.limits["XbarS"])
    for name, value in limits.items():
        subgroup[name + "_mm"] = value
    first, summaries = {}, {}
    for method, paths, unit in iter_chart_paths(pool, np.arange(500), models):
        tau = int(qc.first_alarm(paths, models.limits[method])[0])
        first[method] = tau * unit if tau <= paths.shape[1] else None
        summaries[method] = qc.fault_delay_summary(paths, models.limits[method], 200 // unit, unit)
        alarms = paths[0] > models.limits[method]
        if method == "XbarS":
            subgroup["alarm"] = alarms
            subgroup["first_alarm"] = subgroup["sheet_end"] == first[method]
        else:
            sheet[method + "_alarm"] = alarms
            sheet[method + "_first_alarm"] = sheet["sheet"] == first[method]
        del paths
    ct = qc.cusum_table(pool["loss"], models.loss_mu, models.loss_sigma, models.k, models.limits["CUSUM"])
    et = qc.ewma_table(pool["loss"], models.loss_mu, models.loss_sigma, models.lam, models.limits["EWMA"])
    sheet["cusum_upper"] = ct["upper"]
    sheet["ewma_loss"] = et["ewma"]
    sheet["ewma_startup_score"] = et["score"]
    sheet.to_csv(out / "sheet_trace.csv", index=False)
    subgroup.to_csv(out / "subgroup_trace.csv", index=False)
    ct.to_csv(out / "cusum_table.csv", index=False)
    et.to_csv(out / "ewma_table.csv", index=False)
    np.savez_compressed(out / "example_displacements.npz", U=U,
                        measurement_coordinates=measurement_grid(), change_sheet=np.array(201))
    _plot_trace(out, sheet, subgroup, ct, et, models, first)
    return {"sheets": 500, "change_after_sheet": 200, "first_changed_sheet": 201,
            "fault_family": "broad_Uz_mean", "toy_coefficient": 1.0,
            "first_alarm_sheet": first, "first_alarm_delay_evaluation": summaries,
            "continues_after_alarm_without_reset": True,
            "lower_CUSUM_and_EWMA_table_columns": "diagnostic only; lower alarms are not evaluated"}


def _plot_trace(out, sheet, subgroup, ct, et, models, first):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context({"font.size": 9, "axes.grid": True, "grid.alpha": 0.2}):
        fig, axes = plt.subplots(4, 2, figsize=(15, 13), sharex=True, constrained_layout=True)
        ax = axes.ravel()
        specs = [
            ("XbarS", "Xbar (joint mean/SD OR)", subgroup["sheet_end"], subgroup["xbar_mm"], "mm"),
            ("XbarS", "S (joint mean/SD OR)", subgroup["sheet_end"], subgroup["s_mm"], "mm"),
            ("Hotelling", "Directional XYZ T² — fixed node 44 (zero-based)", sheet["sheet"], sheet["hotelling_T2"], "dimensionless"),
            ("PCA", "PCA T² (joint T²/Q OR)", sheet["sheet"], sheet["pca_T2"], "dimensionless"),
            ("PCA", "PCA Q (joint T²/Q OR)", sheet["sheet"], sheet["pca_Q"], "standardized residual sum of squares"),
            ("OneClassSVM", "RBF one-class SVM: negative decision function", sheet["sheet"], sheet["svm_score"], "decision-score units"),
            ("CUSUM", "Upper loss CUSUM (k = 0.5)", ct["sheet"], ct["upper"], "standardized cumulative loss"),
            ("EWMA", "Upper startup EWMA (λ = 0.1)", et["sheet"], et["ewma"], "loss"),
        ]
        ml = models.chart.moment_limits(models.limits["XbarS"])
        boundaries = [(ml["xbar_lcl"], ml["xbar_ucl"]), (ml["s_lcl"], ml["s_ucl"]),
                      (models.limits["Hotelling"],), (models.r * models.limits["PCA"],),
                      (models.pca.q_scale * models.limits["PCA"],),
                      (models.limits["OneClassSVM"],), (models.limits["CUSUM"],)]
        for i, (method, title, x, y, ylabel) in enumerate(specs):
            a = ax[i]
            a.plot(x, y, color="#176b87", linewidth=0.8)
            if i < 7:
                for j, bound in enumerate(boundaries[i]):
                    a.axhline(bound, color="#c0392b", linestyle="--", label="calibrated limit" if j == 0 else None)
            else:
                a.plot(et["sheet"], et["ucl"], color="#c0392b", linestyle="--", label="startup UCL")
                a.axhline(models.loss_mu, color="gray", linewidth=0.7)
            if i in (0, 1):
                a.axhline(ml["xbar_cl" if i == 0 else "s_cl"], color="gray", linewidth=0.7)
            a.axvline(201, color="#8054a2", linestyle="--", label="change at sheet 201")
            if first[method] is not None:
                a.axvline(first[method], color="#d88000", linestyle=":", label="first alarm")
            alarm_label = str(first[method]) if first[method] is not None else "none by 500"
            a.set_title(f"{title}\nFirst alarm evaluation: {alarm_label}")
            a.set_ylabel(ylabel)
            a.set_xlim(1, 500)
            a.legend(loc="upper left", fontsize=7)
        for a in axes[-1]:
            a.set_xlabel("Sheet (Xbar/S plotted at subgroup end)")
        fig.suptitle("Toy displacement fields — frozen limits, first-alarm evaluation\n"
                     "Trace continues after alarms without reset; toy coefficient 1 is not a standardized or physical shift")
        fig.savefig(out / "control_charts_demo.png", dpi=160)
        plt.close(fig)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_external(paths, n=5):
    archives, records = [], []
    resolved = [Path(p).expanduser().resolve() for p in paths]
    if len(set(resolved)) != 3:
        raise ValueError("Use three distinct, independently obtained fit/cal/test archives")
    for path in resolved:
        if path.suffix.lower() != ".npz":
            raise ValueError("External inputs must be NPZ archives (no pickle)")
        with np.load(path, allow_pickle=False) as archive:
            if "U" not in archive:
                raise ValueError(f"Missing U in {path}")
            U = _displacements(archive["U"])
            if len(U) % n:
                raise ValueError("Every external split must contain a multiple of n sheets")
            reward = archive["reward"] if "reward" in archive else None
            if reward is not None:
                _loss(np.zeros(len(U)), reward, 3.0)
            coordinates = archive["measurement_coordinates"] if "measurement_coordinates" in archive else None
            if coordinates is not None:
                coordinates = np.asarray(coordinates, dtype=float)
                if (coordinates.ndim != 2 or coordinates.shape[0] != U.shape[1]
                        or coordinates.shape[1] not in (2, 3) or not np.isfinite(coordinates).all()):
                    raise ValueError("measurement_coordinates must be finite (G, 2) or (G, 3)")
            archives.append((U, reward, coordinates))
            records.append({"path": str(path), "sha256": _sha256(path), "sheets": len(U),
                            "sites": U.shape[1], "reward_present": reward is not None,
                            "coordinates_present": coordinates is not None})
    if len({item[0].shape[1] for item in archives}) != 1:
        raise ValueError("All external splits must use the same measurement-site count G")
    if len({item[1] is not None for item in archives}) != 1:
        raise ValueError("reward must occur in all three archives or none")
    if len({item[2] is not None for item in archives}) != 1:
        raise ValueError("measurement_coordinates must occur in all three archives or none")
    coordinates = archives[0][2]
    if coordinates is not None and any(not np.array_equal(coordinates, item[2]) for item in archives[1:]):
        raise ValueError("measurement_coordinates must match exactly, including node order")
    if len({record["sha256"] for record in records}) != 3:
        raise ValueError("Identical archives cannot be independent fit/cal/validation splits")
    return archives, records


def _model_metadata(models):
    pca = models.pca
    return {
        "fit_size": models.fit_size, "measurement_sites": models.grid_size,
        "critical_node_zero_based": models.critical_node, "n": models.n, "r": models.r,
        "loss": {"definition": "1 - reward" if models.uses_reward else "clip(max(abs(Uz))/deformation_scale, 0, 1)",
                 "deformation_scale_mm": models.deformation_scale,
                 "mu": models.loss_mu, "sigma_ddof1": models.loss_sigma},
        "XbarS": {"mu_mm": models.chart.mu, "sigma_mm": models.chart.sigma,
                  "sigma_estimator": "mean within-subgroup sample SD / c4(n)",
                  "score": "max(abs standardized subgroup mean, abs standardized subgroup SD)",
                  "limits_mm": models.chart.moment_limits(models.limits["XbarS"])},
        "Hotelling": {"mean_xyz_mm": models.hotelling.mean,
                      "covariance_mm_squared": models.hotelling.covariance,
                      "reference_count": models.hotelling.reference_count},
        "PCA": {"standardize": True, "mean_mm": pca.mean, "scale_mm": pca.scale,
                "retained_feature_mask": pca.mask, "components": pca.pca.components_,
                "pca_internal_mean": pca.pca.mean_, "explained_variance": pca.pca.explained_variance_,
                "explained_variance_ratio": pca.pca.explained_variance_ratio_, "q_scale": pca.q_scale,
                "score": "max(T2/r, Q/training_mean_Q)",
                "T2_limit": models.r * models.limits["PCA"],
                "Q_limit": pca.q_scale * models.limits["PCA"]},
        "OneClassSVM": {"kernel": "rbf", "nu": 0.05, "gamma": 1.0 / models.r,
                        "input": "training-standardized Uz PCA scores / sqrt(training eigenvalues)",
                        "score": "-decision_function; zero is NOT the calibrated limit",
                        "support_vectors": len(models.svm.support_), "offset": models.svm.offset_,
                        "fit_status": models.svm.fit_status_},
        "CUSUM": {"k": models.k, "initial_state": 0, "side": "upper", "reset_after_alarm": False},
        "EWMA": {"lambda": models.lam, "initial_loss_state": models.loss_mu,
                 "initial_standardized_state": 0, "side": "upper",
                 "startup_variance": "sigma^2 * lambda/(2-lambda) * (1-(1-lambda)^(2*t))"},
        "score_limits": models.limits,
    }


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)}")


def _parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("qc_demo_output"))
    parser.add_argument("--fit-size", type=int, default=1500, help="synthetic baseline sheets (multiple of 5)")
    parser.add_argument("--pool-size", type=int, default=20000, help="sheets per synthetic cal/validation/fault pool (multiple of 5)")
    parser.add_argument("--replications", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=4000, help="sheets per trajectory (multiple of 5)")
    parser.add_argument("--target-sheets", type=float, default=370.0)
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--fit-file", type=Path)
    parser.add_argument("--cal-file", type=Path)
    parser.add_argument("--test-file", type=Path, help="independent in-control validation, not a fault archive")
    parser.add_argument("--data-id", help="required external-data provenance identifier")
    return parser


def _validate_args(args):
    for name in ("fit_size", "pool_size", "replications", "horizon"):
        _positive_integer(getattr(args, name), name)
    for name in ("fit_size", "pool_size", "horizon"):
        if getattr(args, name) % 5:
            raise ValueError(f"{name} must be divisible by n=5")
    _positive_integer(args.seed, "seed", 0)
    if not np.isfinite(args.target_sheets) or not 5 < args.target_sheets < args.horizon:
        raise ValueError("target-sheets must exceed n=5 and be below horizon")
    supplied = [args.fit_file is not None, args.cal_file is not None, args.test_file is not None]
    if any(supplied) and not all(supplied):
        raise ValueError("Supply --fit-file, --cal-file and --test-file together, or none")
    if all(supplied) and not (args.data_id and args.data_id.strip()):
        raise ValueError("External mode requires a nonempty --data-id provenance string")
    return all(supplied)


def run(args):
    """Run independent calibration/validation; never refit on subsequent pools."""
    external = _validate_args(args)
    out = args.out.expanduser().resolve()
    if any((out / name).exists() for name in OUTPUTS):
        raise ValueError("Output files already exist; choose a fresh --out directory to avoid mixing experiments")
    streams = _seed_plan(args.seed)
    records = None
    coordinates = None
    if external:
        archives, records = _load_external((args.fit_file, args.cal_file, args.test_file))
        (fit_U, fit_reward, coordinates), (cal_U, cal_reward, _), (val_U, val_reward, _) = archives
        models = fit_models(fit_U, reward=fit_reward)
        cal_pool = score_pool(cal_U, models, reward=cal_reward)
        val_pool = score_pool(val_U, models, reward=val_reward)
        del archives, fit_U, cal_U, val_U
    else:
        fit_U = synthetic_sheets(args.fit_size, np.random.default_rng(streams["fit"]))
        models = fit_models(fit_U)
        del fit_U
        coordinates = measurement_grid()
        cal_pool = score_pool(synthetic_sheets(args.pool_size, np.random.default_rng(streams["cal_pool"])), models)
        val_pool = score_pool(synthetic_sheets(args.pool_size, np.random.default_rng(streams["validation_pool"])), models)
    split_sizes = {"fit": models.fit_size, "calibration": len(cal_pool["d"]), "validation": len(val_pool["d"])}
    loss_diagnostics = {name: {"mean": float(pool["loss"].mean()),
                               "fraction_at_zero": float(np.mean(pool["loss"] == 0)),
                               "fraction_at_one": float(np.mean(pool["loss"] == 1))}
                        for name, pool in (("calibration", cal_pool), ("validation", val_pool))}
    cal = _calibrate(cal_pool, _indices(len(cal_pool["d"]), args, streams["cal_paths"]), models, args.target_sheets)
    del cal_pool
    validation = _validate(val_pool, _indices(len(val_pool["d"]), args, streams["validation_paths"]), models)
    del val_pool
    out.mkdir(parents=True, exist_ok=True)
    cal.to_csv(out / "calibration.csv", index=False)
    validation.to_csv(out / "validation.csv", index=False)
    emitted = ["calibration.csv", "validation.csv"]
    trace = None
    fault_censoring = None
    if not external:
        faults = _fault_experiments(args, streams, models)
        faults.to_csv(out / "fault_comparison.csv", index=False)
        fault_censoring = float(faults["censored_fraction"].max())
        trace = save_trace(out, streams, models)
        emitted.extend(OUTPUTS[2:-1])
    notices = [
        "No matched population ARL claim: thresholds target finite-path empirical restricted means; the objective is a step function.",
        "MCSE is conditional on fitted models, frozen thresholds, and empirical source pools; it omits training, pool, and threshold-selection uncertainty.",
        "Calibration MCSE is descriptive after threshold selection, not an independent uncertainty estimate.",
        "Right-censored runs are capped at H sheets, not H+1. Restricted means are not uncensored ARLs or unrestricted mean delays.",
        "Even zero observed censoring does not prove a negligible population tail; inspect horizon, MCSE, target error and independent validation.",
        "Bootstrap trajectories assume IID comparable sheets; dependence, drift, and nonexchangeability require a different design.",
        "Methods share sheet indices within each experiment; comparisons are paired, not independent method samples.",
        "Individual chart calibration does not calibrate a combined six-chart OR alarm.",
    ]
    if validation["censored_fraction"].max() > 0:
        notices.append("Validation contains censored runs; increase horizon before interpreting a mean as an ARL.")
    if fault_censoring is not None and fault_censoring > 0:
        notices.append("Some toy fault runs are censored; fault delays are restricted means over the reported horizon.")
    if external:
        notices.append("External IID-bootstrap assumption is unverified. No automatic physical, unit, independence, baseline, or provenance validation is performed.")
        if coordinates is None:
            notices.append("External coordinates are absent: equal G does not establish common measurement sites or node ordering; user must verify these.")
    else:
        notices.append("Toy fault coefficients 0.5, 1, 2 are arbitrary amplitude multipliers, not standardized shifts or physical faults.")
    metadata = {
        "mode": "external_IID_bootstrap" if external else "synthetic_toy",
        "data_id": args.data_id if external else "toy_signed_xyz_rank3_v1",
        "runtime": {"python": sys.version, "interpreter": sys.executable, "prefix": sys.prefix,
                    "platform": platform.platform(), "machine": platform.machine(),
                    "libraries": {name: version(name) for name in ("numpy", "scipy", "pandas", "scikit-learn", "matplotlib")}},
        "arguments": vars(args), "external_archives_fit_cal_validation": records,
        "split_sizes": split_sizes, "source_split_policy": "disjoint independently generated streams" if not external else "user-supplied independent in-control archives; independence not verified",
        "seed": args.seed, "rng": "numpy.default_rng / PCG64 / SeedSequence",
        "streams": {name: {"entropy": stream.entropy, "spawn_key": stream.spawn_key,
                            "used": not external or name in ("cal_paths", "validation_paths")}
                    for name, stream in streams.items()},
        "resampling": {"with_replacement": True, "common_indices_across_methods": True,
                       "replications": args.replications, "horizon_sheets": args.horizon,
                       "path_initialization": "zero-state; no alarm resets; full trajectories",
                       "subgroups": "nonoverlapping consecutive n sheets along each trajectory time axis"},
        "units": {"U": "signed XYZ millimetres", "deformation": "max(abs(Uz)) across fixed measurement sites, mm",
                  "loss": "dimensionless [0,1]", "run_lengths_and_delays": "sheets",
                  "XbarS_updates": "5 sheets, alarm assigned to group end", "other_updates": "one sheet",
                  "measurement_coordinates": "external units supplied by data provenance; not inferred" if external else "dimensionless normalized planar x,y on fixed 10x10 measurement grid (not fixture grid)"},
        "measurement_coordinates": coordinates,
        "models": _model_metadata(models), "loss_pool_diagnostics": loss_diagnostics,
        "calibration": {"target_sheets": args.target_sheets, "objective": "restricted first-alarm mean",
                        "max_censoring": 0.01, "bisection_iterations_max": 60,
                        "strict_exceedance": True, "independent_validation": True, "future_refits": False},
        "faults": None if external else {"families": FAULTS, "coefficients": COEFFICIENTS,
                                         "change_index": 0, "pool_size_per_scenario": args.pool_size,
                                         "broad_Uz_increment_mm": "0.15 * coefficient at every node",
                                         "Ux_increment_mm": "0.15 * coefficient at every node",
                                         "local_Uz_increment_mm": "0.18 * coefficient * exp(-((x-.45)^2+(y+.35)^2)/(2*.17^2))",
                                         "max_censored_fraction": fault_censoring},
        "synthetic_generator": None if external else {"independent_XYZ_draws": True,
            "modes": "1, x/RMS(x), y/RMS(y)", "latent_sd_mm_by_XYZ": [[.12, .08, .06], [.10, .07, .09], [.16, .12, .10]],
            "residual_iid_sd_mm": .035, "Uz_mean_mm": "0.45 + 0.08*cos(pi*x)*cos(pi*y)",
            "covariance": "positive residual variance at every node/component; full-rank population covariance"},
        "trace": trace,
        "assumptions": [
            "Fit, calibration, and validation are independent in-control sheet samples from a comparable operating regime.",
            "Sheet observations are IID within each source pool; resampling is not a time-dependence model.",
            "Each row uses the same fixed measurement sites and signed XYZ components in millimetres.",
            "Subgroups are distinct consecutive sheet draws, not repeated nodes on a single sheet; bootstrap draws may repeat source sheets.",
            "XbarS uses normal-theory c4 and SD scaling as score conventions; empirical path calibration, not normality, determines the limit.",
            "Loss is a bounded deformation proxy or user-provided terminal reward complement, not automatically a validated feasibility or quality measure.",
            "All feature selection, normalization, PCA, SVM and loss moments are trained only once on the fit split.",
        ],
        "warnings": notices,
        "deployment_status": "research starter; metadata does not serialize a deployment-ready SVM/model artifact",
        "investigation_policy": "evaluation stops at first alarm; investigate before any operational reset or baseline change; plotted traces continue only for illustration",
        "source_sha256": {name: _sha256(Path(__file__).with_name(name))
                  for name in ("demo.py", "qc_monitoring.py", "requirements.txt")},
        "outputs": emitted + ["run_metadata.json"],
        "external_omissions": list(OUTPUTS[2:-1]) if external else [],
        "external_size_options": "fit-size/pool-size are synthetic-only; all external rows used" if external else None,
    }
    with (out / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, default=_json_default, allow_nan=False)
        handle.write("\n")
    print(f"Wrote {len(emitted) + 1} outputs to {out}")
    print("Limits target empirical restricted means, not certified matched ARLs. See metadata warnings.")
    return metadata


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())