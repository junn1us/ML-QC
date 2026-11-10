"""Fresh Phase I/II comparison; no fits or thresholds use Phase II data.

Scores are memoryless on independent generated sheets. Therefore ARL=1/p
and P(signal by h)=1-(1-p)**h exactly for a frozen rule. Intervals estimate
the conditional simulation probability, not uncertainty over baseline fits.
"""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import beta

if __package__:
    from .qc_monitoring import PCAMonitor, upper_limit
else:
    from qc_monitoring import PCAMonitor, upper_limit

ROOT = Path(__file__).resolve().parent
METHODS = ("PCA_T2_Q", "Shewhart_d")


def binomial_summary(k, n, confidence=.95, horizon=50):
    """Two-sided Clopper-Pearson interval with monotone transformations."""
    if not isinstance(k, (int, np.integer)) or not isinstance(n, (int, np.integer)) or not 0 <= k <= n or n < 1:
        raise ValueError("Require integer 0 <= k <= n and n > 0")
    if not 0 < confidence < 1 or horizon < 1:
        raise ValueError("Invalid confidence or horizon")
    a = (1 - confidence) / 2
    lo = float(beta.ppf(a, k, n - k + 1)) if k else 0.0
    hi = float(beta.ppf(1 - a, k + 1, n - k)) if k < n else 1.0
    p = k / n
    pd_h = lambda q: float(-np.expm1(horizon * np.log1p(-q))) if q < 1 else 1.0
    return {"sheets": n, "alarms": k, "p_hat": p, "p_lo": lo, "p_hi": hi,
            "arl_hat": 1 / p if p else None, "arl_lo": 1 / hi,
            "arl_hi": 1 / lo if lo else None, "confidence": confidence,
            "detection_by_h": pd_h(p), "detection_by_h_lo": pd_h(lo),
            "detection_by_h_hi": pd_h(hi), "detection_horizon": horizon}


def grid():
    x, y = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
    return x.ravel(), y.ravel()


def generate_profile(n, rng, family=None, coefficient=0.0):
    """Same Uz marginal law as demo.synthetic_sheets, without unused XYZ draws.

    New independent random streams, not identical numerical sheets. x,y are
    normalized measurement coordinates. This is not a mechanical simulation.
    """
    x, y = grid()
    modes = np.column_stack((np.ones(100), x / np.sqrt(np.mean(x*x)), y / np.sqrt(np.mean(y*y))))
    coefficients = rng.normal(size=(n, 3)) * [.16, .12, .10]
    z = coefficients @ modes.T + rng.normal(0, .035, (n, 100))
    z += .45 + .08 * np.cos(np.pi * x) * np.cos(np.pi * y)
    if family == "broad_Uz_mean":
        z += .15 * coefficient
    elif family == "local_Uz_distortion":
        z += .18 * coefficient * np.exp(-((x - .45)**2 + (y + .35)**2) / (2 * .17**2))
    elif family is not None:
        raise ValueError("Unknown fault family")
    return z


def score(profile, pca, mu, sigma):
    d = np.max(np.abs(profile), axis=1)
    return np.column_stack((pca.joint_score(profile), np.abs((d - mu) / sigma)))


def chunks(n, size):
    for start in range(0, n, size):
        yield start, min(size, n - start)


def event_counts(n, size, rng, pca, mu, sigma, thresholds, family=None, coefficient=0):
    counts = np.zeros(2, dtype=np.int64)
    both = 0
    # Store batch counts as a compact audit trail; never reuse these for fitting.
    batches = []
    for start, count in chunks(n, size):
        values = score(generate_profile(count, rng, family, coefficient), pca, mu, sigma)
        alarm = values > thresholds
        k = alarm.sum(axis=0)
        joint = int(np.all(alarm, axis=1).sum())
        counts += k
        both += joint
        batches.append({"start_sheet": start + 1, "sheets": count, "PCA_alarms": int(k[0]),
                        "Shewhart_alarms": int(k[1]), "both_alarms": joint})
    return counts, both, batches


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "iise_report/experiment_plan.json")
    parser.add_argument("--out", type=Path, default=ROOT / "iise_report/results")
    args = parser.parse_args(argv)
    plan_bytes = args.plan.read_bytes()
    plan = json.loads(plan_bytes)
    if args.out.exists():
        raise ValueError("Use a fresh output directory; do not overwrite Phase II results")
    args.out.mkdir(parents=True)
    (args.out / "locked_plan.json").write_bytes(plan_bytes)
    streams = np.random.SeedSequence(plan["root_seed"]).spawn(3 + len(plan["fault_scenarios"]))
    rngs = [np.random.default_rng(s) for s in streams]
    fit = generate_profile(plan["fit_sheets"], rngs[0])
    pca = PCAMonitor.fit(fit, r=plan["pca_rank"], standardize=plan["pca_standardize"])
    d = np.max(np.abs(fit), axis=1)
    mu, sigma = float(d.mean()), float(d.std(ddof=1))
    size, nc = plan["chunk_size"], plan["calibration_sheets"]
    calibration = np.empty((nc, 2))
    print("Phase I fitted; generating held-out calibration sheets", flush=True)
    for start, count in chunks(nc, size):
        calibration[start:start + count] = score(generate_profile(count, rngs[1]), pca, mu, sigma)
    thresholds = np.array([upper_limit(calibration[:, j], 1 / plan["target_arl0_sheets"]) for j in range(2)])
    cal_k = np.sum(calibration > thresholds, axis=0)
    rank = int(np.ceil((nc + 1) * (1 - 1 / plan["target_arl0_sheets"])))
    np.savez_compressed(args.out / "frozen_model.npz", profile_mean=pca.mean, profile_scale=pca.scale,
                        mask=pca.mask, loadings=pca.pca.components_, pca_internal_mean=pca.pca.mean_,
                        eigenvalues=pca.pca.explained_variance_, q_scale=pca.q_scale,
                        scalar_mu=mu, scalar_sigma=sigma, thresholds=thresholds)
    frozen = {"thresholds": dict(zip(METHODS, thresholds.tolist())), "scalar_mu": mu,
              "scalar_sigma": sigma, "scalar_limits_mm": [mu - thresholds[1] * sigma, mu + thresholds[1] * sigma],
              "pca_T2_limit": float(pca.r * thresholds[0]), "pca_Q_limit": float(pca.q_scale * thresholds[0]),
              "q_scale": pca.q_scale, "calibration_rank_1based": rank,
              "phase1_tail_counts": dict(zip(METHODS, cal_k.tolist())),
              "plan_sha256": hashlib.sha256(plan_bytes).hexdigest()}
    frozen_path = args.out / "frozen_before_phase2.json"
    frozen_path.write_text(json.dumps(frozen, indent=2) + "\n")
    frozen_hash = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    del calibration, fit
    print("Limits frozen and saved; opening fresh Phase II in-control stream", thresholds, flush=True)
    n = plan["phase2_in_control_sheets"]
    counts, both, batches = event_counts(n, size, rngs[2], pca, mu, sigma, thresholds)
    pd.DataFrame(batches).to_csv(args.out / "phase2_ic_batch_counts.csv", index=False)
    validation = [{"method": method, **binomial_summary(int(k), n, plan["marginal_confidence"], plan["detection_horizon_sheets"])}
                  for method, k in zip(METHODS, counts)]
    pd.DataFrame(validation).to_csv(args.out / "phase2_validation.csv", index=False)
    # Bonferroni: two 97.5% marginal CP intervals provide >=95% simultaneous coverage.
    simultaneous_conf = 1 - (1 - plan["simultaneous_arl_confidence"]) / 2
    sim = [binomial_summary(int(k), n, simultaneous_conf) for k in counts]
    ratio_hat = validation[1]["p_hat"] / validation[0]["p_hat"]
    ratio_lo, ratio_hi = sim[1]["p_lo"] / sim[0]["p_hi"], sim[1]["p_hi"] / sim[0]["p_lo"]
    tol, target = plan["matching_relative_tolerance"], plan["target_arl0_sheets"]
    match = all(v["arl_hi"] is not None and v["arl_lo"] >= target*(1-tol) and v["arl_hi"] <= target*(1+tol) for v in sim)
    match = bool(match and ratio_lo >= 1-tol and ratio_hi <= 1+tol)
    comparison = {"match_criterion_passed": match, "arl_ratio_PCA_over_Shewhart": ratio_hat,
                  "simultaneous_ratio_lo": ratio_lo, "simultaneous_ratio_hi": ratio_hi,
                  "simultaneous_marginal_intervals": dict(zip(METHODS, sim)),
                  "both_alarm_count": both, "neither_alarm_count": int(n - counts.sum() + both),
                  "target_tolerance_sheets": [target*(1-tol), target*(1+tol)]}
    print("Phase II validation:", validation, "Matching:", match, flush=True)
    rows = []
    for idx, scenario in enumerate(plan["fault_scenarios"]):
        family, coefficient = scenario["family"], scenario["coefficient"]
        nf = plan["fault_sheets_per_scenario"]
        counts, joint, batches = event_counts(nf, size, rngs[idx+3], pca, mu, sigma, thresholds, family, coefficient)
        slug = f"{family}_{coefficient:g}"
        pd.DataFrame(batches).to_csv(args.out / f"fault_{slug}_batch_counts.csv", index=False)
        for method, k in zip(METHODS, counts):
            rows.append({**scenario, "method": method, "both_alarm_count": joint,
                         "amplitude_parameter_mm": coefficient * (.15 if family == "broad_Uz_mean" else .18),
                         **binomial_summary(int(k), nf, plan["marginal_confidence"], plan["detection_horizon_sheets"])})
        print("Fault", slug, "counts", counts, flush=True)
    pd.DataFrame(rows).to_csv(args.out / "phase2_faults.csv", index=False)
    assert hashlib.sha256(frozen_path.read_bytes()).hexdigest() == frozen_hash
    result = {"plan": plan, "frozen_phase1": frozen, "phase2_comparison": comparison,
              "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                for name in ("matched_phase_study.py", "qc_monitoring.py")},
              "streams": [{"role": role, "entropy": s.entropy, "spawn_key": s.spawn_key}
                          for role, s in zip(["PhaseI_fit", "PhaseI_calibration", "PhaseII_in_control"] +
                                             [f"PhaseII_fault_{i}" for i in range(len(rows)//2)], streams)],
              "runtime": {"python": sys.version, "executable": sys.executable,
                          "libraries": {name: version(name) for name in ("numpy", "scipy", "scikit-learn", "pandas")}},
              "inference_warning": "Conditional fixed-model IID probabilities. ARLs inferred geometrically, not observed censored run averages. No baseline-fitting uncertainty or physical validation.",
              "independence": "Fresh generator draws per sheet; no empirical-pool resampling; common sheets across methods, separate fit/cal/test/fault streams",
              "no_adaptive_recalibration": True}
    (args.out / "study_metadata.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("Completed:", args.out, flush=True)


if __name__ == "__main__":
    main()