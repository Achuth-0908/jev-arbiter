"""Calculation of error correlation and effective independent votes."""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple


def compute_pairwise_error_correlations(
    agent_predictions: np.ndarray,
    ground_truth: np.ndarray,
) -> Tuple[np.ndarray, float, float]:
    """Compute pairwise Matthews/Pearson correlation of errors across k agents.
    
    Args:
        agent_predictions: Binary array of shape (k, N) where k is number of agents, N is claims.
        ground_truth: Binary array of shape (N,) with true labels.
        
    Returns:
        corr_matrix: (k, k) array of pairwise error correlations.
        mean_phi: Average off-diagonal correlation phi_bar.
        n_eff: Effective independent votes k / (1 + (k - 1) * phi_bar).
    """
    k, n = agent_predictions.shape
    if k <= 1:
        return np.ones((1, 1)), 0.0, 1.0

    # Error residual matrix: 1 if agent erred, 0 if correct
    errors = (agent_predictions != ground_truth[np.newaxis, :]).astype(float)

    corr_matrix = np.eye(k)
    correlations = []

    for i in range(k):
        for j in range(i + 1, k):
            e_i = errors[i]
            e_j = errors[j]

            std_i = np.std(e_i)
            std_j = np.std(e_j)

            if std_i > 1e-8 and std_j > 1e-8:
                corr = np.corrcoef(e_i, e_j)[0, 1]
                if np.isnan(corr):
                    corr = 0.0
            else:
                corr = 0.0

            # Correlation clipped to [-1, 1]
            corr = float(np.clip(corr, -1.0, 1.0))
            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr
            correlations.append(corr)

    # Average off-diagonal correlation phi_bar
    mean_phi = float(np.mean(correlations)) if correlations else 0.0
    # Effective independent votes
    phi_clamped = max(0.0, min(1.0, mean_phi))
    n_eff = float(k / (1.0 + (k - 1) * phi_clamped))

    return corr_matrix, mean_phi, n_eff


def calculate_discounted_agreement(
    votes_for: int,
    total_agents: int,
    mean_phi: float,
) -> float:
    """Calculate agreement ratio discounted by error correlation.
    
    If agents are highly correlated, unanimous agreement carries less effective weight.
    """
    if total_agents <= 0:
        return 0.0
    phi_clamped = max(0.0, min(1.0, mean_phi))
    n_eff = total_agents / (1.0 + (total_agents - 1) * phi_clamped)
    raw_ratio = votes_for / total_agents

    # Effective vote weighting
    effective_agreement = (raw_ratio * n_eff) / total_agents
    return float(np.clip(effective_agreement, 0.0, 1.0))
