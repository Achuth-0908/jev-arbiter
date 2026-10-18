"""Calibration and classification evaluation metrics."""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import roc_auc_score


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute standard Brier Score = (1/N) * sum((y_prob - y_true)^2)."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return float(np.mean((y_prob - y_true) ** 2))


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> float:
    """Compute Expected Calibration Error (ECE).
    
    ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)
    if n == 0:
        return 0.0

    if strategy == "quantile":
        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_edges = np.percentile(y_prob, quantiles * 100)
        bin_edges[0] = 0.0
        bin_edges[-1] = 1.0001
    else:
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0
    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob >= low) & (y_prob < high if i < n_bins - 1 else y_prob <= high)
        bin_size = np.sum(mask)
        if bin_size > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            ece += (bin_size / n) * abs(bin_acc - bin_conf)

    return float(ece)


def maximum_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Maximum Calibration Error (MCE) = max_m |acc(B_m) - conf(B_m)|."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    max_err = 0.0
    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob >= low) & (y_prob < high if i < n_bins - 1 else y_prob <= high)
        if np.sum(mask) > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            err = abs(bin_acc - bin_conf)
            if err > max_err:
                max_err = err

    return float(max_err)


def auroc_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Area Under Receiver Operating Characteristic Curve."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(np.unique(y_true)) < 2:
        return 0.50
    return float(roc_auc_score(y_true, y_prob))


def reliability_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Tuple[List[float], List[float], List[float], List[int]]:
    """Compute coordinates for reliability diagrams (calibration curves).
    
    Returns:
        (bin_centers, bin_accuracies, bin_confidences, bin_counts)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    centers, accs, confs, counts = [], [], [], []
    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob >= low) & (y_prob < high if i < n_bins - 1 else y_prob <= high)
        bin_count = int(np.sum(mask))
        center = float((low + high) / 2.0)
        centers.append(center)
        counts.append(bin_count)
        if bin_count > 0:
            accs.append(float(np.mean(y_true[mask])))
            confs.append(float(np.mean(y_prob[mask])))
        else:
            accs.append(center)
            confs.append(center)

    return centers, accs, confs, counts


def selective_accuracy(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_coverage: float = 0.80,
) -> Tuple[float, float, float]:
    """Calculate accuracy at a fixed coverage percentile based on prediction confidence distance from 0.5.
    
    Returns:
        (selective_accuracy, threshold, actual_coverage)
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    confidence_margin = np.abs(y_prob - 0.5)

    cutoff_idx = int(np.floor((1.0 - target_coverage) * len(y_prob)))
    sorted_indices = np.argsort(confidence_margin)
    selected_indices = sorted_indices[cutoff_idx:]

    if len(selected_indices) == 0:
        return 0.0, 0.5, 0.0

    selected_preds = (y_prob[selected_indices] >= 0.5).astype(int)
    acc = float(np.mean(selected_preds == y_true[selected_indices]))
    thresh = float(confidence_margin[sorted_indices[cutoff_idx]])
    cov = float(len(selected_indices) / len(y_prob))

    return acc, thresh, cov
