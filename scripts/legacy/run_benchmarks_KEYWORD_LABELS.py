"""LEGACY - DO NOT USE FOR RESULTS.

Ground-truth labels in this script are hardcoded keyword matches (e.g. "oppenheimer" in claim) on 3 toy
questions, and it scores with a mock verifier. Its numbers measure nothing. Kept only for reference;
use `python -m src.runs phase1` for real, leakage-safe, pre-registered evaluation.
"""
"""Benchmark and metrics comparison runner across arbitration strategies."""

from __future__ import annotations

import asyncio
import time
from typing import Dict, List
import numpy as np
import pandas as pd

from src.clients.cache import CacheStore
from src.clients.jev import JevDecisionsClient
from src.data.qa_loader import get_sample_multihop_instances
from src.fuse.engine import ArbitrationEngine
from src.verify.verifier import JevClaimVerifier
from src.calibrate.metrics import brier_score, expected_calibration_error


async def run_metrics_benchmark(is_live: bool = False) -> pd.DataFrame:
    """Executes a side-by-side empirical benchmark of all arbitration strategies."""
    cache = CacheStore()
    client = JevDecisionsClient(mock_mode=not is_live, cache_store=cache)
    verifier = JevClaimVerifier(client=client)
    engine = ArbitrationEngine(
        verifier=verifier,
        acceptance_threshold=0.65,
        abstention_threshold=0.35,
        default_mean_phi=0.30,
    )

    instances = get_sample_multihop_instances()
    strategies = [
        ("1. Single Agent (Claude)", "single_agent"),
        ("2. Naive Majority Vote", "majority_vote"),
        ("3. Self-Confidence Weighted", "self_confidence"),
        ("4. Raw Jev Argmax", "raw_jev"),
        ("5. Calibrated Jev Alone", "calibrated_jev"),
        ("6. Calibrated Fusion (Proposed)", "calibrated_fusion"),
    ]

    records = []

    for label, strat in strategies:
        total_claims_evaluated = 0
        total_accepted = 0
        correct_accepted = 0
        hallucinations_leaked = 0
        all_true_labels = []
        all_predicted_scores = []
        total_cost = 0.0
        latencies = []

        for inst in instances:
            t0 = time.time()
            if strat == "single_agent":
                # Single agent: Claude alone
                candidate = {"agent_claude": inst.candidate_responses["agent_claude"]}
                res = await engine.arbitrate(
                    query=inst.question,
                    candidate_responses=candidate,
                    reference_documents=inst.passages,
                    strategy="majority_vote",
                )
            else:
                res = await engine.arbitrate(
                    query=inst.question,
                    candidate_responses=inst.candidate_responses,
                    candidate_confidences=inst.candidate_confidences,
                    reference_documents=inst.passages,
                    strategy=strat if strat in ("majority_vote", "self_confidence", "raw_jev", "calibrated_jev") else "calibrated_fusion",
                )

            lat = (time.time() - t0) * 1000.0
            latencies.append(lat)
            total_cost += res.total_cost_usd

            for ac in res.arbitrated_claims:
                total_claims_evaluated += 1
                # Ground truth check against gold answer
                is_fact = (
                    ("oppenheimer" in ac.canonical_claim.lower() and "berkeley" in ac.canonical_claim.lower()) or
                    ("c++" in ac.canonical_claim.lower() and "1998" in ac.canonical_claim.lower()) or
                    ("ribosome" in ac.canonical_claim.lower() and "protein" in ac.canonical_claim.lower()) or
                    ("lack" in ac.canonical_claim.lower() or "not have" in ac.canonical_claim.lower() or "non-membrane" in ac.canonical_claim.lower())
                ) and not ("stanford" in ac.canonical_claim.lower() or "polio" in ac.canonical_claim.lower() or "2011" in ac.canonical_claim.lower() or "atp" in ac.canonical_claim.lower())

                label_val = 1 if is_fact else 0
                score_val = (
                    ac.fused_prob if strat == "calibrated_fusion"
                    else (ac.calibrated_jev_prob if strat == "calibrated_jev"
                    else (ac.raw_jev_prob if strat == "raw_jev"
                    else (ac.raw_agreement_ratio * ac.mean_self_confidence if strat == "self_confidence"
                    else ac.raw_agreement_ratio)))
                )

                all_true_labels.append(label_val)
                all_predicted_scores.append(score_val)

                if ac.status == "ACCEPTED":
                    total_accepted += 1
                    if is_fact:
                        correct_accepted += 1
                    else:
                        hallucinations_leaked += 1

        accuracy = (correct_accepted / total_accepted) if total_accepted > 0 else 0.0
        leak_rate = (hallucinations_leaked / total_accepted) if total_accepted > 0 else 0.0
        bs = brier_score(np.array(all_true_labels), np.array(all_predicted_scores))
        ece = expected_calibration_error(np.array(all_true_labels), np.array(all_predicted_scores), n_bins=5)
        mean_lat = float(np.mean(latencies))
        mean_cost = total_cost / max(1, len(instances))

        records.append({
            "Strategy": label,
            "Factual Precision": f"{accuracy * 100:.1f}%",
            "Hallucination Leakage": f"{leak_rate * 100:.1f}%",
            "Brier Score": f"{bs:.4f}",
            "ECE": f"{ece:.4f}",
            "Avg Latency": f"{mean_lat:.1f} ms",
            "Cost / Query": f"${mean_cost:.5f}",
        })

    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("EMPIRICAL BENCHMARK COMPARISON: WITHOUT JEV vs. WITH JEV")
    print("=" * 80 + "\n")
    df_metrics = asyncio.run(run_metrics_benchmark(is_live=False))
    print(df_metrics.to_string(index=False))
    print("\n" + "=" * 80 + "\n")
