"""Tests for error correlation and effective independent votes."""

import numpy as np
import pytest

from src.fuse.effective_votes import (
    compute_pairwise_error_correlations,
    calculate_discounted_agreement,
)


def test_independent_agents_full_votes():
    # 3 agents making completely uncorrelated errors
    # Agent 0 fails on claim 0
    # Agent 1 fails on claim 1
    # Agent 2 fails on claim 2
    ground_truth = np.array([1, 1, 1, 1, 1, 1])
    preds = np.array([
        [0, 1, 1, 1, 1, 1],
        [1, 0, 1, 1, 1, 1],
        [1, 1, 0, 1, 1, 1],
    ])

    corr, phi_bar, n_eff = compute_pairwise_error_correlations(preds, ground_truth)
    assert phi_bar <= 0.05
    # When phi_bar ~ 0, n_eff ~ k (3)
    assert abs(n_eff - 3.0) < 0.5


def test_perfectly_correlated_agents_collapse_to_one():
    # 4 agents making identical errors
    ground_truth = np.array([1, 1, 1, 1])
    preds = np.array([
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [0, 0, 1, 1],
    ])

    corr, phi_bar, n_eff = compute_pairwise_error_correlations(preds, ground_truth)
    assert phi_bar == pytest.approx(1.0, abs=1e-5)
    # When phi_bar == 1, n_eff == 1.0
    assert n_eff == pytest.approx(1.0, abs=1e-5)


def test_discounted_agreement_bounds():
    # 3 agents all vote yes, with phi_bar = 0.8
    disc_agr = calculate_discounted_agreement(votes_for=3, total_agents=3, mean_phi=0.8)
    # N_eff for k=3, phi=0.8 is 3 / (1 + 2 * 0.8) = 3 / 2.6 = 1.1538
    # discounted ratio = 1.1538 / 3 = 0.3846
    assert 0.0 <= disc_agr <= 1.0
    assert disc_agr < 0.50  # Significantly discounted due to high correlation
