"""Frozen-baseline control charts. One observation is one independent sheet.

All alarms use strict exceedance. Sequential helpers start at the in-control
state: pass a complete trajectory, not separate one-row batches for live use.
"""
from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from scipy import linalg, special, stats
from sklearn.decomposition import PCA


def _array(x, ndim=None, name="data"):
    a = np.asarray(x, dtype=float)
    if a.size == 0 or not np.isfinite(a).all():
        raise ValueError(f"{name} must be nonempty and finite")
    if ndim is not None and a.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    return a


def _integer(value, minimum=1, name="count"):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _probability(value, name="alpha"):
    if not np.isfinite(value) or not 0 < value < 1:
        raise ValueError(f"{name} must lie strictly between zero and one")


def _scale(mu, sigma):
    if not np.isfinite(mu) or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("mu must be finite and sigma strictly positive")


def _smoothing(lam):
    if not np.isfinite(lam) or not 0 < lam <= 1:
        raise ValueError("lam must be in (0, 1]")


def _nonnegative(value, name):
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


def extract_features(U, critical_node=44):
    """U is (sheets, fixed measurement sites, signed XYZ), in millimetres."""
    U = _array(U, 3, "U")
    if U.shape[2] != 3:
        raise ValueError("U must have shape (N, G, 3); do not fabricate missing components")
    node = _integer(critical_node, 0, "critical_node")
    if node >= U.shape[1]:
        raise ValueError("critical_node is outside the measurement grid")
    return np.max(np.abs(U[:, :, 2]), axis=1), U[:, node, :].copy(), U[:, :, 2].copy()


def subgroup(y, n=5):
    """Group distinct consecutive comparable sheets; never group spatial nodes."""
    y = _array(y, 1)
    n = _integer(n, 2, "n")
    if y.size % n:
        raise ValueError("Incomplete subgroup: buffer remaining sheets; do not drop them")
    return y.reshape(-1, n)


def c4(n):
    n = _integer(n, 2, "n")
    return float(np.sqrt(2 / (n - 1)) * np.exp(special.gammaln(n / 2) - special.gammaln((n - 1) / 2)))


@dataclass
class XbarS:
    n: int
    mu: float
    sigma: float

    def __post_init__(self):
        self.n = _integer(self.n, 2, "n")
        _scale(self.mu, self.sigma)

    @classmethod
    def fit(cls, y, n=5):
        groups = subgroup(y, n)
        return cls(n, float(groups.mean()), float(groups.std(axis=1, ddof=1).mean() / c4(n)))

    def statistics(self, groups):
        groups = _array(groups, name="groups")
        if groups.ndim < 2 or groups.shape[-1] != self.n:
            raise ValueError("groups must end in a subgroup-size axis")
        return groups.mean(axis=-1), groups.std(axis=-1, ddof=1)

    def joint_score(self, groups):
        xbar, s = self.statistics(groups)
        correction = c4(self.n)
        mean_score = np.abs(xbar - self.mu) / (self.sigma / np.sqrt(self.n))
        sd_score = np.abs(s - correction * self.sigma) / (self.sigma * np.sqrt(1 - correction**2))
        return np.maximum(mean_score, sd_score)

    def moment_limits(self, width=3.0):
        _nonnegative(width, "width")
        correction = c4(self.n)
        delta = width * self.sigma / np.sqrt(self.n)
        s_delta = width * self.sigma * np.sqrt(1 - correction**2)
        return {"xbar_cl": self.mu, "xbar_lcl": self.mu - delta, "xbar_ucl": self.mu + delta,
                "s_cl": correction * self.sigma, "s_lcl": max(0.0, correction * self.sigma - s_delta),
                "s_ucl": correction * self.sigma + s_delta}

    def probability_limits(self, alpha_x, alpha_s):
        _probability(alpha_x, "alpha_x")
        _probability(alpha_s, "alpha_s")
        delta = stats.norm.ppf(1 - alpha_x / 2) * self.sigma / np.sqrt(self.n)
        sd_limits = self.sigma * np.sqrt(stats.chi2.ppf([alpha_s / 2, 1 - alpha_s / 2], self.n - 1) / (self.n - 1))
        return {"xbar_cl": self.mu, "xbar_lcl": self.mu - delta, "xbar_ucl": self.mu + delta,
                "s_cl": c4(self.n) * self.sigma, "s_lcl": float(sd_limits[0]), "s_ucl": float(sd_limits[1])}


@dataclass
class Hotelling:
    mean: np.ndarray
    covariance: np.ndarray
    reference_count: int | None = None

    def __post_init__(self):
        self.mean = _array(self.mean, 1, "mean").copy()
        self.covariance = _array(self.covariance, 2, "covariance").copy()
        p = self.mean.size
        if self.covariance.shape != (p, p) or not np.allclose(self.covariance, self.covariance.T):
            raise ValueError("Covariance must be square, symmetric, and match the mean")
        if np.linalg.cond(self.covariance) > 1e12:
            raise ValueError("Singular or ill-conditioned covariance; remove redundant coordinates")
        try:
            self._chol = linalg.cholesky(self.covariance, lower=True)
        except linalg.LinAlgError as exc:
            raise ValueError("Covariance must be positive definite") from exc
        if self.reference_count is not None:
            self.reference_count = _integer(self.reference_count, p + 1, "reference_count")

    @classmethod
    def fit(cls, x):
        x = _array(x, 2)
        if x.shape[0] <= x.shape[1]:
            raise ValueError("Need more baseline observations than coordinates")
        return cls(x.mean(axis=0), np.atleast_2d(np.cov(x, rowvar=False, ddof=1)), len(x))

    def whiten(self, x):
        x = _array(x, 2)
        if x.shape[1] != self.mean.size:
            raise ValueError("Feature count differs from the frozen baseline")
        return linalg.solve_triangular(self._chol, (x - self.mean).T, lower=True).T

    def score(self, x):
        return np.sum(self.whiten(x)**2, axis=1)

    def analytic_limit(self, alpha=1 / 370, phase=2):
        _probability(alpha)
        if phase not in (1, 2):
            raise ValueError("phase must be 1 (fitted observations) or 2 (independent future observations)")
        p, m = self.mean.size, self.reference_count
        if m is None:
            return float(stats.chi2.ppf(1 - alpha, p))
        if phase == 1:
            if m <= p + 1:
                raise ValueError("Phase I beta limit requires m > p + 1")
            return float((m - 1)**2 / m * stats.beta.ppf(1 - alpha, p / 2, (m - p - 1) / 2))
        return float(p * (m + 1) * (m - 1) / (m * (m - p)) * stats.f.ppf(1 - alpha, p, m - p))


@dataclass
class PCAMonitor:
    mean: np.ndarray
    scale: np.ndarray
    mask: np.ndarray
    pca: PCA
    q_scale: float

    @property
    def r(self):
        return self.pca.components_.shape[0]

    @classmethod
    def fit(cls, profile, r=3, standardize=True):
        x = _array(profile, 2)
        if len(x) < 3:
            raise ValueError("Need at least three baseline profiles")
        r = _integer(r, 1, "r")
        mean, sd = x.mean(axis=0), x.std(axis=0, ddof=1)
        # A training-only mask is applied unchanged to every later observation.
        mask = sd > np.finfo(float).eps * np.maximum(1, np.max(np.abs(x), axis=0))
        if r >= min(len(x), int(mask.sum())):
            raise ValueError("r must leave a nonempty residual subspace after removing constant coordinates")
        scale = np.where(mask, sd, 1.0) if standardize else np.ones(x.shape[1])
        z = ((x - mean) / scale)[:, mask]
        pca = PCA(n_components=r, svd_solver="full").fit(z)
        if pca.explained_variance_.min() <= 1e-12 * pca.explained_variance_.max():
            raise ValueError("Retained PCA eigenvalues are degenerate")
        residual = z - pca.inverse_transform(pca.transform(z))
        q_scale = float(np.mean(np.sum(residual**2, axis=1)))
        if q_scale <= np.finfo(float).eps * max(1.0, float(np.mean(np.sum(z**2, axis=1)))):
            raise ValueError("No residual variation; joint T2/Q calibration is undefined")
        return cls(mean, scale, mask, pca, q_scale)

    def transform(self, profile):
        x = _array(profile, 2)
        if x.shape[1] != self.mean.size:
            raise ValueError("Observation-grid size differs from training")
        return ((x - self.mean) / self.scale)[:, self.mask]

    def whitened_scores(self, profile):
        return self.pca.transform(self.transform(profile)) / np.sqrt(self.pca.explained_variance_)

    def scores(self, profile):
        z = self.transform(profile)
        a = self.pca.transform(z)
        residual = z - self.pca.inverse_transform(a)
        return np.sum(a**2 / self.pca.explained_variance_, axis=1), np.sum(residual**2, axis=1)

    def joint_score(self, profile):
        t2, q = self.scores(profile)
        return np.maximum(t2 / self.r, q / self.q_scale)


def loss_from_reward(reward):
    reward = _array(reward, 1, "terminal reward")
    if np.any((reward < 0) | (reward > 1)):
        raise ValueError("Terminal reward must be in [0,1]; specify a fixed transformation otherwise")
    return 1 - reward


def bounded_loss(d, deformation_scale=3.0, feasibility_loss=None, weight=1.0):
    d = _array(d, 1, "absolute deformation")
    _scale(0, deformation_scale)
    if np.any(d < 0) or not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("Deformation must be nonnegative and weight in [0,1]")
    deformation = np.clip(d / deformation_scale, 0, 1)
    if feasibility_loss is None:
        if weight != 1:
            raise ValueError("A feasibility loss is required when weight != 1")
        return deformation
    feasibility = _array(feasibility_loss, 1, "feasibility_loss")
    if feasibility.shape != d.shape or np.any((feasibility < 0) | (feasibility > 1)):
        raise ValueError("Feasibility loss must match d and lie in [0,1]")
    return weight * deformation + (1 - weight) * feasibility


def cusum_score_paths(y, mu, sigma, k=0.5):
    y = _array(y, 2, "paths")
    _scale(mu, sigma)
    _nonnegative(k, "k")
    z = (y - mu) / sigma
    scores, state = np.empty_like(z), np.zeros(len(z))
    for t in range(z.shape[1]):
        state = np.maximum(0, state + z[:, t] - k)
        scores[:, t] = state
    return scores


def cusum_table(y, mu, sigma, k=0.5, h=5.0):
    y = _array(y, 1)
    _scale(mu, sigma)
    _nonnegative(k, "k")
    _nonnegative(h, "h")
    z = (y - mu) / sigma
    upper = cusum_score_paths(y[None, :], mu, sigma, k)[0]
    lower = cusum_score_paths((-y)[None, :], -mu, sigma, k)[0]
    up_run, lo_run = np.zeros(len(y), dtype=int), np.zeros(len(y), dtype=int)
    for t in range(len(y)):
        up_run[t] = (up_run[t - 1] if t else 0) + 1 if upper[t] > 0 else 0
        lo_run[t] = (lo_run[t - 1] if t else 0) + 1 if lower[t] > 0 else 0
    return pd.DataFrame({"sheet": np.arange(1, len(y) + 1), "loss": y, "z": z,
                         "upper_increment": z - k, "lower_increment": -z - k,
                         "upper": upper, "lower": lower, "upper_run": up_run, "lower_run": lo_run,
                         "h": h, "upper_alarm": upper > h, "lower_alarm": lower > h})


def ewma_score_paths(y, mu, sigma, lam=0.1):
    y = _array(y, 2, "paths")
    _scale(mu, sigma)
    _smoothing(lam)
    state, scores = np.zeros(len(y)), np.empty_like(y)
    z = (y - mu) / sigma
    for t in range(y.shape[1]):
        state = lam * z[:, t] + (1 - lam) * state
        factor = np.sqrt(lam / (2 - lam) * (1 - (1 - lam)**(2 * (t + 1))))
        scores[:, t] = state / factor
    return scores


def ewma_table(y, mu, sigma, lam=0.1, L=3.0):
    y = _array(y, 1)
    _nonnegative(L, "L")
    score = ewma_score_paths(y[None, :], mu, sigma, lam)[0]
    t = np.arange(1, len(y) + 1)
    sd = sigma * np.sqrt(lam / (2 - lam) * (1 - (1 - lam)**(2 * t)))
    return pd.DataFrame({"sheet": t, "loss": y, "ewma": mu + score * sd,
                         "center": mu, "startup_sd": sd, "ucl": mu + L * sd,
                         "lcl": mu - L * sd, "score": score, "upper_alarm": score > L})


def ma_score(y, mu, sigma, window=5, two_sided=True):
    y = _array(y, 1)
    _scale(mu, sigma)
    window = _integer(window, 1, "window")
    t = np.arange(1, len(y) + 1)
    count = np.minimum(t, window)
    csum = np.concatenate(([0.0], np.cumsum(y)))
    means = (csum[t] - csum[np.maximum(0, t - window)]) / count
    score = (means - mu) / (sigma / np.sqrt(count))
    return np.abs(score) if two_sided else score


def mewma_score(x, hotelling, lam=0.1):
    _smoothing(lam)
    z = hotelling.whiten(x)
    state, scores = np.zeros(z.shape[1]), np.empty(len(z))
    for t, row in enumerate(z):
        state = lam * row + (1 - lam) * state
        factor = lam / (2 - lam) * (1 - (1 - lam)**(2 * (t + 1)))
        scores[t] = np.dot(state, state) / factor
    return scores


def upper_limit(scores, alpha):
    """Held-out finite-rank threshold; marginal exchangeability, not an ARL claim."""
    scores = _array(scores, 1, "scores")
    _probability(alpha)
    rank = math.ceil((len(scores) + 1) * (1 - alpha))
    if rank > len(scores):
        raise ValueError("Calibration sample too small for a finite threshold at this alpha")
    return float(np.partition(scores, rank - 1)[rank - 1])


def first_alarm(paths, limit):
    """Return 1-based first alarm updates; H+1 denotes a censored run."""
    paths = _array(paths, 2, "paths")
    if not np.isfinite(limit):
        raise ValueError("limit must be finite")
    exceed = paths > limit
    return np.where(exceed.any(axis=1), exceed.argmax(axis=1) + 1, paths.shape[1] + 1)


def run_length_summary(paths, limit, unit_sheets=1):
    unit_sheets = _integer(unit_sheets, 1, "unit_sheets")
    paths = _array(paths, 2, "paths")
    tau = first_alarm(paths, limit)
    horizon = paths.shape[1]
    observed = tau <= horizon
    capped = np.minimum(tau, horizon) * unit_sheets
    mean = float(capped.mean())
    return {"replications": len(paths), "horizon_sheets": horizon * unit_sheets,
            "unit_sheets": unit_sheets, "signals": int(observed.sum()),
            "censored_fraction": float(np.mean(~observed)), "restricted_mean_sheets": mean,
            "restricted_mean_mcse": float(capped.std(ddof=1) / np.sqrt(len(capped))) if len(capped) > 1 else None,
            "arl_sheets": mean if observed.all() else None}


def calibrate_run_length(paths, target_sheets=370, unit_sheets=1, max_censoring=0.01, iterations=60):
    """Search monotone empirical restricted mean, with fixed paths throughout.

    Finite paths give a step function. This does not certify population ARL or
    account for uncertainty in baseline fitting or the empirical source pool.
    """
    paths = _array(paths, 2, "paths")
    unit_sheets = _integer(unit_sheets, 1, "unit_sheets")
    iterations = _integer(iterations, 1, "iterations")
    if not np.isfinite(target_sheets) or not unit_sheets < target_sheets < paths.shape[1] * unit_sheets:
        raise ValueError("Target must exceed one update and be below the horizon in sheets")
    if not np.isfinite(max_censoring) or not 0 <= max_censoring <= 1:
        raise ValueError("max_censoring must be in [0,1]")
    lo, hi = float(np.nextafter(paths.min(), -np.inf)), float(paths.max())
    best_h, best_gap = lo, float("inf")
    for candidate in (lo, hi):
        gap = abs(run_length_summary(paths, candidate, unit_sheets)["restricted_mean_sheets"] - target_sheets)
        if gap < best_gap:
            best_h, best_gap = candidate, gap
    for _ in range(iterations):
        mid = lo + (hi - lo) / 2
        if mid == lo or mid == hi:
            break
        summary = run_length_summary(paths, mid, unit_sheets)
        estimate = summary["restricted_mean_sheets"]
        gap = abs(estimate - target_sheets)
        if gap < best_gap:
            best_h, best_gap = mid, gap
        if estimate < target_sheets:
            lo = mid
        else:
            hi = mid
    result = run_length_summary(paths, best_h, unit_sheets)
    if result["censored_fraction"] > max_censoring:
        raise ValueError("Excessive calibration censoring: increase the horizon")
    result.update(target_sheets=float(target_sheets), absolute_target_error=best_gap, limit=best_h)
    return best_h, result


def fault_delay_summary(paths, limit, change_index=0, unit_sheets=1):
    """Change follows change_index updates; condition delay on no earlier alarm.

    For a subgroup chart, change_index counts complete groups, not sheets.
    No survivor means conditional quantities are undefined, not zero.
    """
    paths = _array(paths, 2, "paths")
    change_index = _integer(change_index, 0, "change_index")
    unit_sheets = _integer(unit_sheets, 1, "unit_sheets")
    if change_index >= paths.shape[1]:
        raise ValueError("Change must precede the end of the observation horizon")
    tau = first_alarm(paths, limit)
    survive = tau > change_index
    event = tau[survive] <= paths.shape[1]
    delay = (np.minimum(tau[survive], paths.shape[1]) - change_index) * unit_sheets
    return {"replications": len(paths), "survivors": int(survive.sum()),
            "change_index_updates": change_index,
            "early_alarm_fraction": float(np.mean(~survive)),
            "post_change_horizon_sheets": (paths.shape[1] - change_index) * unit_sheets,
            "detection_probability": float(event.mean()) if event.size else None,
            "censored_fraction": float(np.mean(~event)) if event.size else None,
            "restricted_delay_sheets": float(delay.mean()) if delay.size else None,
            "delay_mcse": float(delay.std(ddof=1) / np.sqrt(delay.size)) if delay.size > 1 else None,
            "mean_delay_sheets": float(delay.mean()) if delay.size and event.all() else None}