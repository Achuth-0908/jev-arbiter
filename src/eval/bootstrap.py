"""Cluster-bootstrap confidence intervals and paired tests (resampling whole clusters).

Claims that share a source document / question are correlated, so the resampling unit is the
cluster, never the individual claim. Seeds default to ``bootstrap_seed`` in configs/seeds.yaml.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np

from ..config import load_seeds

MetricFn = Callable[[np.ndarray, np.ndarray], float]


def _resolve_seed(seed: Optional[int]) -> int:
    return load_seeds()["bootstrap_seed"] if seed is None else int(seed)


class _Clusters:
    """Precomputed grouping so each resample costs O(n) instead of O(n * clusters)."""

    def __init__(self, item_ids: Sequence) -> None:
        ids = np.asarray(item_ids)
        self.order = np.argsort(ids, kind="stable")
        _, self.starts, self.counts = np.unique(ids[self.order], return_index=True, return_counts=True)
        self.n = len(self.starts)

    def sample(self, rng: np.random.RandomState) -> np.ndarray:
        picks = rng.randint(0, self.n, size=self.n)
        return np.concatenate([self.order[self.starts[c] : self.starts[c] + self.counts[c]] for c in picks])


def cluster_bootstrap_multi(
    item_ids: Sequence,
    y_true: Sequence,
    y_score: Sequence,
    metric_fns: Dict[str, MetricFn],
    n_bootstraps: int = 1000,
    ci_level: float = 0.95,
    seed: Optional[int] = None,
    skip_single_class: bool = True,
) -> Dict[str, Tuple[float, float, float]]:
    """All metrics on the *same* cluster resamples. Returns {name: (point, lo, hi)}.

    Resamples containing a single class are skipped (AUROC is undefined there).
    """
    y = np.asarray(y_true)
    s = np.asarray(y_score)
    rng = np.random.RandomState(_resolve_seed(seed))
    clusters = _Clusters(item_ids)

    points = {name: float(fn(y, s)) for name, fn in metric_fns.items()}
    draws: Dict[str, List[float]] = {name: [] for name in metric_fns}
    for _ in range(n_bootstraps):
        idx = clusters.sample(rng)
        yb = y[idx]
        if skip_single_class and (yb.min() == yb.max()):
            continue
        sb = s[idx]
        for name, fn in metric_fns.items():
            draws[name].append(float(fn(yb, sb)))

    lo_q, hi_q = (1.0 - ci_level) / 2.0 * 100.0, (1.0 - (1.0 - ci_level) / 2.0) * 100.0
    out: Dict[str, Tuple[float, float, float]] = {}
    for name in metric_fns:
        d = draws[name]
        if d:
            out[name] = (points[name], float(np.percentile(d, lo_q)), float(np.percentile(d, hi_q)))
        else:
            out[name] = (points[name], float("nan"), float("nan"))
    return out


def cluster_bootstrap_ci(
    item_ids: Sequence,
    y_true: Sequence,
    y_pred: Sequence,
    metric_fn: MetricFn,
    n_bootstraps: int = 1000,
    ci_level: float = 0.95,
    seed: Optional[int] = None,
) -> Tuple[float, float, float]:
    """Single-metric cluster bootstrap CI. Returns (point_estimate, ci_lower, ci_upper)."""
    res = cluster_bootstrap_multi(
        item_ids, y_true, y_pred, {"m": metric_fn}, n_bootstraps, ci_level, seed, skip_single_class=False
    )
    return res["m"]


def paired_cluster_bootstrap_test(
    item_ids: Sequence,
    y_true: Sequence,
    y_pred_a: Sequence,
    y_pred_b: Sequence,
    metric_fn: MetricFn,
    n_bootstraps: int = 1000,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """One-sided paired test that strategy B beats A on ``metric_fn`` (higher = better).

    Returns (observed_difference_B_minus_A, p_value) where p = (#{diff <= 0} + 1) / (B + 1).
    """
    y = np.asarray(y_true)
    a = np.asarray(y_pred_a)
    b = np.asarray(y_pred_b)
    rng = np.random.RandomState(_resolve_seed(seed))
    clusters = _Clusters(item_ids)

    observed = float(metric_fn(y, b) - metric_fn(y, a))
    diffs = []
    for _ in range(n_bootstraps):
        idx = clusters.sample(rng)
        diffs.append(float(metric_fn(y[idx], b[idx]) - metric_fn(y[idx], a[idx])))
    p_val = (float(np.sum(np.asarray(diffs) <= 0.0)) + 1.0) / (n_bootstraps + 1.0)
    return observed, p_val
