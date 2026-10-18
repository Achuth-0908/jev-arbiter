"""Recalibration models: Temperature Scaling and Isotonic Regression."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression


class TemperatureScaler:
    """Parametric Platt / Temperature scaling: p_cal = sigmoid(logit(p) / T)."""

    def __init__(self, clip_bounds: tuple[float, float] = (0.01, 0.99)):
        self.temperature: float = 1.0
        self.clip_bounds = clip_bounds
        self.fitted: bool = False

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> "TemperatureScaler":
        """Fit temperature parameter T on calibration split via NLL minimization."""
        y_prob = np.clip(np.asarray(y_prob, dtype=float), self.clip_bounds[0], self.clip_bounds[1])
        y_true = np.asarray(y_true, dtype=float)

        logits = logit(y_prob)

        def nll_objective(t: float) -> float:
            if t <= 0.01:
                return 1e9
            scaled_logits = logits / t
            probs = expit(scaled_logits)
            probs = np.clip(probs, 1e-12, 1.0 - 1e-12)
            # Binary Cross Entropy
            loss = -np.mean(y_true * np.log(probs) + (1.0 - y_true) * np.log(1.0 - probs))
            return float(loss)

        res = minimize_scalar(nll_objective, bounds=(0.05, 10.0), method="bounded")
        self.temperature = float(res.x)
        self.fitted = True
        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        """Apply fitted temperature scaling."""
        y_prob = np.clip(np.asarray(y_prob, dtype=float), self.clip_bounds[0], self.clip_bounds[1])
        if not self.fitted or abs(self.temperature - 1.0) < 1e-6:
            return y_prob

        logits = logit(y_prob)
        scaled_logits = logits / self.temperature
        recalibrated = expit(scaled_logits)
        return np.clip(recalibrated, self.clip_bounds[0], self.clip_bounds[1])


class IsotonicCalibrator:
    """Non-parametric monotonic Isotonic Regression calibrator."""

    def __init__(self, clip_bounds: tuple[float, float] = (0.01, 0.99)):
        self.clip_bounds = clip_bounds
        self.model = IsotonicRegression(
            y_min=clip_bounds[0],
            y_max=clip_bounds[1],
            out_of_bounds="clip",
        )
        self.fitted: bool = False

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> "IsotonicCalibrator":
        """Fit isotonic regression on calibration split."""
        y_prob = np.clip(np.asarray(y_prob, dtype=float), self.clip_bounds[0], self.clip_bounds[1])
        y_true = np.asarray(y_true, dtype=float)
        self.model.fit(y_prob, y_true)
        self.fitted = True
        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        """Transform uncalibrated probabilities using fitted isotonic curve."""
        y_prob = np.clip(np.asarray(y_prob, dtype=float), self.clip_bounds[0], self.clip_bounds[1])
        if not self.fitted:
            return y_prob
        transformed = self.model.predict(y_prob)
        return np.clip(transformed, self.clip_bounds[0], self.clip_bounds[1])
