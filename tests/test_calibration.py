"""Tests for calibration metrics and scaling models."""

import numpy as np
import pytest

from src.calibrate.metrics import (
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
    auroc_score,
    reliability_curve,
    selective_accuracy,
)
from src.calibrate.scalers import TemperatureScaler, IsotonicCalibrator


def test_brier_and_ece_perfect_calibration():
    # Perfectly calibrated predictions
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([1.0, 1.0, 0.0, 0.0])

    bs = brier_score(y_true, y_prob)
    assert bs == 0.0

    ece = expected_calibration_error(y_true, y_prob, n_bins=10)
    assert ece == 0.0

    auc = auroc_score(y_true, y_prob)
    assert auc == 1.0


def test_temperature_scaler_reduces_miscalibration():
    # Overconfident uncalibrated predictions
    rng = np.random.RandomState(42)
    n = 200
    y_true = rng.binomial(1, 0.6, size=n)
    # Predict extreme probabilities (overconfident)
    raw_probs = np.where(y_true == 1, 0.95, 0.05)

    initial_ece = expected_calibration_error(y_true, raw_probs, n_bins=5)
    
    scaler = TemperatureScaler()
    scaler.fit(raw_probs, y_true)
    cal_probs = scaler.transform(raw_probs)
    
    post_ece = expected_calibration_error(y_true, cal_probs, n_bins=5)
    assert scaler.fitted
    assert post_ece <= initial_ece


def test_isotonic_calibrator_monotonicity():
    rng = np.random.RandomState(42)
    y_prob = np.linspace(0.05, 0.95, 50)
    y_true = (y_prob > 0.50).astype(int)

    iso = IsotonicCalibrator()
    iso.fit(y_prob, y_true)
    transformed = iso.transform(y_prob)

    # Transformed output must be non-decreasing
    assert np.all(np.diff(transformed) >= -1e-6)
    assert np.all(transformed >= 0.01)
    assert np.all(transformed <= 0.99)
