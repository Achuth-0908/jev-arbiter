"""Cluster-bootstrap statistical confidence intervals and paired hypothesis testing."""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple
import numpy as np


def cluster_bootstrap_ci(
    item_ids: List[str],
    y_true: List[int],
    y_pred: List[int | float],
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstraps: int = 1000,
    ci_level: float = 0.95,
    seed: int = 1337,
) -> Tuple[float, float, float]:
    """Compute cluster bootstrap confidence interval grouped by question/item ID.
    
    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(seed)
    unique_items = np.unique(item_ids)
    n_clusters = len(unique_items)

    item_to_indices = {item: np.where(np.array(item_ids) == item)[0] for item in unique_items}

    # Point estimate
    point_est = float(metric_fn(np.array(y_true), np.array(y_pred)))

    bootstrap_scores = []
    for _ in range(n_bootstraps):
        sampled_clusters = rng.choice(unique_items, size=n_clusters, replace=True)
        sampled_indices = []
        for cluster in sampled_clusters:
            sampled_indices.extend(item_to_indices[cluster])

        sampled_indices = np.array(sampled_indices)
        sub_true = np.array(y_true)[sampled_indices]
        sub_pred = np.array(y_pred)[sampled_indices]

        score = metric_fn(sub_true, sub_pred)
        bootstrap_scores.append(score)

    alpha = 1.0 - ci_level
    lower_pct = (alpha / 2.0) * 100.0
    upper_pct = (1.0 - alpha / 2.0) * 100.0

    ci_lower = float(np.percentile(bootstrap_scores, lower_pct))
    ci_upper = float(np.percentile(bootstrap_scores, upper_pct))

    return point_est, ci_lower, ci_upper


def paired_cluster_bootstrap_test(
    item_ids: List[str],
    y_true: List[int],
    y_pred_a: List[int | float],
    y_pred_b: List[int | float],
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstraps: int = 1000,
    seed: int = 1337,
) -> Tuple[float, float]:
    """Paired cluster bootstrap hypothesis test to check if Strategy B significantly outperforms Strategy A.
    
    Returns:
        (difference, p_value)
    """
    rng = np.random.RandomState(seed)
    unique_items = np.unique(item_ids)
    n_clusters = len(unique_items)
    item_to_indices = {item: np.where(np.array(item_ids) == item)[0] for item in unique_items}

    orig_a = metric_fn(np.array(y_true), np.array(y_pred_a))
    orig_b = metric_fn(np.array(y_true), np.array(y_pred_b))
    orig_diff = orig_b - orig_a

    diffs = []
    for _ in range(n_bootstraps):
        sampled_clusters = rng.choice(unique_items, size=n_clusters, replace=True)
        sampled_indices = []
        for cluster in sampled_clusters:
            sampled_indices.extend(item_to_indices[cluster])

        sampled_indices = np.array(sampled_indices)
        sub_true = np.array(y_true)[sampled_indices]
        sub_a = np.array(y_pred_a)[sampled_indices]
        sub_b = np.array(y_pred_b)[sampled_indices]

        diff = metric_fn(sub_true, sub_b) - metric_fn(sub_true, sub_a)
        diffs.append(diff)

    # Empirical one-sided p-value: fraction of bootstraps where B <= A
    diffs_arr = np.array(diffs)
    p_val = float(np.mean(diffs_arr <= 0.0))

    return float(orig_diff), p_val
