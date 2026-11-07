"""Benchmark evaluation runner across all arbitration strategies."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
from pydantic import BaseModel, Field

from ..calibrate.metrics import brier_score, expected_calibration_error, auroc_score
from ..calibrate.scalers import TemperatureScaler, IsotonicCalibrator
from ..data.loaders import FactClaimInstance, generate_benchmark_synthetic_subset, create_frozen_splits
from ..fuse.engine import ArbitrationEngine
from ..fuse.effective_votes import compute_pairwise_error_correlations
from ..verify.verifier import JevClaimVerifier


class StrategyEvaluationResult(BaseModel):
    """Evaluation summary for a specific arbitration strategy."""
    strategy_name: str
    accuracy: float
    brier: float
    ece: float
    auroc: float
    mean_latency_ms: float
    mean_cost_usd: float
    n_samples: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BenchmarkRunner:
    """Executes systematic comparisons across the arbitration hierarchy."""

    def __init__(self, verifier: JevClaimVerifier):
        self.verifier = verifier

    async def run_phase1_calibration_probe(
        self,
        cal_instances: List[FactClaimInstance],
        test_instances: List[FactClaimInstance],
    ) -> Dict[str, Any]:
        """Phase 1 probe: evaluate raw Jev vs Recalibrated Jev (Temperature & Isotonic) on test split."""
        # 1. Collect Jev probabilities on calibration split
        cal_probs = []
        cal_true = []
        for inst in cal_instances:
            res = await self.verifier.verify(inst.claim, inst.evidence)
            cal_probs.append(res.raw_p_supported)
            cal_true.append(inst.label)

        cal_probs_arr = np.array(cal_probs)
        cal_true_arr = np.array(cal_true)

        # 2. Fit scalers on calibration split ONLY
        temp_scaler = TemperatureScaler().fit(cal_probs_arr, cal_true_arr)
        iso_scaler = IsotonicCalibrator().fit(cal_probs_arr, cal_true_arr)

        # 3. Collect Jev probabilities on held-out test split
        test_probs = []
        test_true = []
        nondeterminism_vars = []
        invariance_diffs = []

        for inst in test_instances:
            res = await self.verifier.verify(inst.claim, inst.evidence, check_invariance=True)
            test_probs.append(res.raw_p_supported)
            test_true.append(inst.label)
            nondeterminism_vars.append(res.variance_across_repeats)
            if res.option_invariance_divergence is not None:
                invariance_diffs.append(res.option_invariance_divergence)

        test_probs_arr = np.array(test_probs)
        test_true_arr = np.array(test_true)

        temp_test_probs = temp_scaler.transform(test_probs_arr)
        iso_test_probs = iso_scaler.transform(test_probs_arr)

        def eval_metrics(probs: np.ndarray) -> Dict[str, float]:
            preds = (probs >= 0.50).astype(int)
            acc = float(np.mean(preds == test_true_arr))
            bs = brier_score(test_true_arr, probs)
            ece = expected_calibration_error(test_true_arr, probs, n_bins=10)
            auc = auroc_score(test_true_arr, probs)
            return {"accuracy": acc, "brier": bs, "ece": ece, "auroc": auc}

        return {
            "fitted_temperature": temp_scaler.temperature,
            "raw_jev": eval_metrics(test_probs_arr),
            "temperature_scaled": eval_metrics(temp_test_probs),
            "isotonic_scaled": eval_metrics(iso_test_probs),
            "mean_repeat_variance": float(np.mean(nondeterminism_vars)),
            "mean_option_invariance_divergence": float(np.mean(invariance_diffs)) if invariance_diffs else 0.0,
            "test_sample_count": len(test_instances),
            "temp_scaler": temp_scaler,
            "iso_scaler": iso_scaler,
        }

    async def run_full_comparison(
        self,
        qa_instances: List[Any],
        label_fn: Callable[[Any, Any], int],
        temp_scaler: Optional[TemperatureScaler] = None,
    ) -> List[StrategyEvaluationResult]:
        """Run comparison across all strategy rungs.

        ``label_fn(qa_instance, arbitrated_claim) -> {0, 1}`` supplies ground truth for each
        claim (1 = factually correct). It is required: this runner never infers labels, and it
        raises rather than substituting placeholder metrics when no labeled claims exist.
        Metrics are pooled over all claims; question-level cluster bootstrap CIs are computed
        downstream from the per-claim records in ``metadata['records']``.
        """
        strategies = [
            ("Majority Vote", "majority_vote"),
            ("Self-Confidence Weighted", "self_confidence"),
            ("Raw Jev Argmax", "raw_jev"),
            ("Calibrated Jev", "calibrated_jev"),
            ("Calibrated Fusion", "calibrated_fusion"),
        ]

        engine = ArbitrationEngine(
            verifier=JevClaimVerifier(
                client=self.verifier.client,
                calibrator=temp_scaler,
            )
        )

        results: List[StrategyEvaluationResult] = []

        for display_name, strat_key in strategies:
            labels: List[int] = []
            scores: List[float] = []
            accepted: List[int] = []
            groups: List[str] = []
            costs: List[float] = []
            latencies: List[float] = []

            for qa in qa_instances:
                arb_res = await engine.arbitrate(
                    query=qa.question,
                    candidate_responses=qa.candidate_responses,
                    candidate_confidences=qa.candidate_confidences,
                    reference_documents=qa.passages,
                    strategy=strat_key,
                )
                costs.append(arb_res.total_cost_usd)
                latencies.append(arb_res.total_latency_ms)

                for ac in arb_res.arbitrated_claims:
                    labels.append(int(label_fn(qa, ac)))
                    scores.append(_strategy_score(strat_key, ac))
                    accepted.append(1 if ac.status == "ACCEPTED" else 0)
                    groups.append(str(qa.id))

            if not labels:
                raise ValueError(f"No labeled claims produced for strategy {display_name!r}")

            y = np.array(labels)
            p = np.array(scores)
            acc_mask = np.array(accepted) == 1
            precision = float(np.mean(y[acc_mask])) if acc_mask.any() else float("nan")

            results.append(
                StrategyEvaluationResult(
                    strategy_name=display_name,
                    accuracy=float(np.mean((p >= 0.5).astype(int) == y)),
                    brier=brier_score(y, p),
                    ece=expected_calibration_error(y, p, n_bins=10),
                    auroc=auroc_score(y, p),
                    mean_latency_ms=float(np.mean(latencies)),
                    mean_cost_usd=float(np.mean(costs)),
                    n_samples=len(labels),
                    metadata={
                        "accepted_precision": precision,
                        "n_accepted": int(acc_mask.sum()),
                        "n_questions": len(qa_instances),
                        "records": {"y": labels, "score": scores, "accepted": accepted, "group": groups},
                    },
                )
            )

        return results


def _strategy_score(strategy: str, ac: Any) -> float:
    """The probability-like score each strategy actually thresholds on."""
    if strategy == "calibrated_fusion":
        return float(ac.fused_prob)
    if strategy == "calibrated_jev":
        return float(ac.calibrated_jev_prob)
    if strategy == "raw_jev":
        return float(ac.raw_jev_prob)
    if strategy == "self_confidence":
        return float(ac.raw_agreement_ratio * ac.mean_self_confidence)
    return float(ac.raw_agreement_ratio)
