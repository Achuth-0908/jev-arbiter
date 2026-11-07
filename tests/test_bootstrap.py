"""Cluster bootstrap: determinism, correct clustering behaviour, and paired-test sanity."""

import numpy as np

from src.calibrate.metrics import auroc_score
from src.eval.bootstrap import (
    cluster_bootstrap_ci,
    cluster_bootstrap_multi,
    paired_cluster_bootstrap_test,
)


def _acc(y, p):
    return float(np.mean((p >= 0.5).astype(int) == y))


def _data(n_clusters=40, per=10, seed=0):
    rng = np.random.RandomState(seed)
    ids, y, p = [], [], []
    for c in range(n_clusters):
        label = int(rng.rand() < 0.5)
        shift = rng.randn() * 0.35  # shared per-cluster error => within-cluster correlation of mistakes
        for _ in range(per):
            ids.append(f"c{c}")
            y.append(label)
            p.append(float(np.clip(0.5 + (0.15 if label else -0.15) + shift + rng.randn() * 0.1, 0.01, 0.99)))
    return ids, np.array(y), np.array(p)


def test_deterministic_for_a_seed_and_differs_across_seeds():
    ids, y, p = _data()
    a = cluster_bootstrap_ci(ids, y, p, _acc, 200, seed=5)
    b = cluster_bootstrap_ci(ids, y, p, _acc, 200, seed=5)
    c = cluster_bootstrap_ci(ids, y, p, _acc, 200, seed=6)
    assert a == b and a != c


def test_interval_contains_point_estimate():
    ids, y, p = _data()
    point, lo, hi = cluster_bootstrap_ci(ids, y, p, _acc, 300, seed=1)
    assert lo <= point <= hi


def test_cluster_ci_is_wider_than_naive_iid_ci_when_claims_are_correlated():
    ids, y, p = _data()
    _, lo_c, hi_c = cluster_bootstrap_ci(ids, y, p, _acc, 400, seed=1)
    singletons = [f"s{i}" for i in range(len(ids))]  # treat every claim as its own cluster
    _, lo_i, hi_i = cluster_bootstrap_ci(singletons, y, p, _acc, 400, seed=1)
    assert (hi_c - lo_c) > 1.5 * (hi_i - lo_i)


def test_multi_metrics_share_resamples_and_match_single():
    ids, y, p = _data()
    multi = cluster_bootstrap_multi(ids, y, p, {"acc": _acc, "auroc": auroc_score}, 200, seed=3)
    single = cluster_bootstrap_ci(ids, y, p, _acc, 200, seed=3)
    assert multi["acc"][0] == single[0]
    assert set(multi) == {"acc", "auroc"}


def test_resamples_with_a_single_class_are_skipped_not_crashing():
    ids = ["a"] * 5 + ["b"] * 5
    y = np.array([1] * 5 + [0] * 5)
    p = np.array([0.9] * 5 + [0.1] * 5)
    out = cluster_bootstrap_multi(ids, y, p, {"auroc": auroc_score}, 100, seed=0)
    point, lo, hi = out["auroc"]
    assert point == 1.0 and not np.isnan(lo)


def test_paired_test_detects_clear_improvement_and_not_noise():
    ids, y, _ = _data(n_clusters=60, seed=2)
    rng = np.random.RandomState(9)
    p_good = np.clip(0.5 + (y - 0.5) * 0.9 + rng.randn(len(y)) * 0.1, 0.01, 0.99)  # strongly informative
    p_bad = rng.rand(len(y))  # uninformative
    diff, pval = paired_cluster_bootstrap_test(ids, y, p_bad, p_good, auroc_score, 300, seed=4)
    assert diff > 0.1 and pval < 0.05
    diff2, pval2 = paired_cluster_bootstrap_test(ids, y, p_good, p_good, auroc_score, 300, seed=4)
    assert diff2 == 0.0 and pval2 > 0.5
