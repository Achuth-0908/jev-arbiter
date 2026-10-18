"""Jev Noul verification templates, repetition management, and option invariance checks."""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from ..clients.jev import JevDecisionsClient
from ..calibrate.scalers import TemperatureScaler, IsotonicCalibrator


class ClaimVerificationResult(BaseModel):
    """Output of verifying a single claim against evidence."""
    claim: str
    evidence_snippet: str
    raw_p_supported: float
    calibrated_p_supported: float
    variance_across_repeats: float
    is_supported: bool
    latency_ms: float
    cost_usd: float
    cached: bool
    option_invariance_divergence: Optional[float] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class JevClaimVerifier:
    """Orchestrates Jev Noul verification with recalibration and robustness testing."""

    DEFAULT_CRITERIA = "Is the claim strictly and directly supported by the provided evidence text?"

    def __init__(
        self,
        client: Optional[JevDecisionsClient] = None,
        calibrator: Optional[TemperatureScaler | IsotonicCalibrator] = None,
        repeats: int = 3,
        clip_bounds: Tuple[float, float] = (0.01, 0.99),
    ):
        self.client = client or JevDecisionsClient()
        self.calibrator = calibrator
        self.repeats = repeats
        self.clip_bounds = clip_bounds

    async def verify(
        self,
        claim: str,
        evidence: str,
        options: Optional[List[str]] = None,
        criteria: Optional[str] = None,
        check_invariance: bool = False,
    ) -> ClaimVerificationResult:
        """Run Jev Noul verification across repeats and apply calibration."""
        options = options or ["Yes", "No"]
        crit = criteria or self.DEFAULT_CRITERIA

        # Primary decision call
        resp = await self.client.evaluate_claim(
            claim=claim,
            evidence=evidence,
            options=options,
            criteria=crit,
            repeats=self.repeats,
            clip_bounds=self.clip_bounds,
        )

        pos_opt = options[0]
        raw_p = resp.clipped_probabilities.get(pos_opt, 0.50)

        # Apply calibration if fitted
        if self.calibrator is not None and getattr(self.calibrator, "fitted", False):
            cal_p = float(self.calibrator.transform(np.array([raw_p]))[0])
        else:
            cal_p = raw_p

        invariance_diff = None
        if check_invariance:
            # Swapped / Alternative wording probe: ["True", "False"]
            alt_options = ["True", "False"]
            alt_resp = await self.client.evaluate_claim(
                claim=claim,
                evidence=evidence,
                options=alt_options,
                criteria=crit,
                repeats=self.repeats,
                clip_bounds=self.clip_bounds,
            )
            alt_p = alt_resp.clipped_probabilities.get(alt_options[0], 0.50)
            invariance_diff = abs(raw_p - alt_p)

        return ClaimVerificationResult(
            claim=claim,
            evidence_snippet=evidence,
            raw_p_supported=raw_p,
            calibrated_p_supported=cal_p,
            variance_across_repeats=resp.variance,
            is_supported=(cal_p >= 0.50),
            latency_ms=resp.latency_ms,
            cost_usd=resp.cost_usd,
            cached=resp.cached,
            option_invariance_divergence=invariance_diff,
            details={"mean_probabilities": resp.mean_probabilities},
        )
